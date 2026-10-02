import json
from datetime import date

import pytest
from sqlalchemy import insert, text

import core
from core import engine, interventions, mask_pii
from tools import PlaybookSearchTool, ReadOnlySQLTool, RecentTicketsTool, RunContext


def test_mask_pii():
    s = mask_pii("Mail priya.sharma@acme.com or call +1 415 555 1234, thanks Priya Sharma", ["Priya Sharma"])
    assert "@" not in s and "415" not in s and "Priya" not in s and "Sharma" not in s


def ctx_for(aid, names):
    return RunContext(run_id="R-T", account_id=aid, contact_names=names)


def test_sql_tool_records_ref_and_masks(seeded, acct):
    aid, name, _ = acct
    ctx = ctx_for(aid, [name])
    out = ReadOnlySQLTool(ctx)._run(f"SELECT feature, SUM(sessions) AS s FROM usage_daily WHERE account_id = '{aid}' GROUP BY feature")
    ref = out.split("\n")[0].split(": ")[1]
    assert ref in ctx.query_refs and ref in ctx.evidence_ids()
    assert ReadOnlySQLTool(ctx)._run("SELECT * FROM playbook").startswith("ERROR")


def test_ticket_tool_masks_pii(seeded):
    with engine.connect() as c:
        aid = c.execute(text("SELECT account_id FROM tickets GROUP BY account_id ORDER BY COUNT(*) DESC LIMIT 1")).scalar()
    names = core.contact_names(aid)
    ctx = ctx_for(aid, names)
    out = RecentTicketsTool(ctx)._run(days=400, limit=25)
    assert ctx.ticket_refs and "@" not in out and "+1 415" not in out
    assert all(n.split()[0] not in out for n in names)


def test_playbook_search_returns_seeded_ids(seeded):
    ctx = ctx_for("A-10000", [])
    out = PlaybookSearchTool(ctx)._run("repeated export bug tickets frustrated after a release", k=3)
    assert "PB-014" in ctx.playbook_refs and "PB-014" in out


# ── Flask review UI ────────────────────────────────────────────────────────────
@pytest.fixture()
def client(seeded, acct):
    import review_app
    from tests.test_pipeline_flow import BODY
    aid = acct[0]
    prof = {"account_id": aid, "risk_score": .9, "risk_tier": "high", "sentiment": "frustrated", "confidence": "high",
            "likely_root_cause": "export bug", "signals": [{"type": "usage", "detail": "down 60%", "evidence": ["Q-aaaaaaaa"]}]}
    iv = {"account_id": aid, "strategy": "Technical escalation", "rationale": "r", "playbook_refs": ["PB-014"],
          "draft_email": {"subject": "Your exports", "body": BODY}}
    with engine.begin() as c:
        c.execute(text("DELETE FROM churn_interventions"))
        rid = c.execute(insert(interventions).values(
            run_id="R-U", run_date=date.today(), account_id=aid, risk_score=.9, risk_tier="high", arr=1000.0,
            profile_json=json.dumps(prof), intervention_json=json.dumps(iv), guardrail_passed=True,
            status="pending_review", final_subject="Your exports", final_body=BODY)).inserted_primary_key[0]
    app = review_app.app
    app.config["TESTING"] = True
    return app.test_client(), rid, BODY


def status_of(rid):
    with engine.connect() as c:
        return c.execute(text("SELECT status FROM churn_interventions WHERE id=:i"), {"i": rid}).scalar()


def test_queue_page_renders(client):
    cl, rid, _ = client
    html = cl.get("/").get_data(as_text=True)
    assert "export bug" in html and "Technical escalation" in html and "Approve" in html
    assert cl.get("/?status=holdout").status_code == 200


def test_approve_requires_reviewer(client):
    cl, rid, body = client
    cl.post(f"/act/{rid}", data={"action": "approve", "subject": "Your exports", "body": body})
    assert status_of(rid) == "pending_review"


def test_approve_unchanged_then_mark_sent(client):
    cl, rid, body = client
    cl.post(f"/act/{rid}", data={"action": "approve", "reviewer": "Sam", "subject": "Your exports", "body": body})
    assert status_of(rid) == "approved"
    assert "Open in mail client" in cl.get("/?status=approved").get_data(as_text=True)
    cl.post(f"/act/{rid}", data={"action": "sent"})
    assert status_of(rid) == "sent"


def test_edit_is_saved_as_edited_and_rechecked(client):
    cl, rid, body = client
    cl.post(f"/act/{rid}", data={"action": "approve", "reviewer": "Sam", "subject": "Your exports", "body": body + "\nPS: we guarantee it"})
    assert status_of(rid) == "pending_review"  # guardrail blocked the edit
    cl.post(f"/act/{rid}", data={"action": "approve", "reviewer": "Sam", "subject": "Your exports", "body": body.replace("Friday", "Thursday")})
    assert status_of(rid) == "edited"


def test_reject_and_add_rule(client):
    cl, rid, _ = client
    cl.post(f"/act/{rid}", data={"action": "reject", "reviewer": "Sam", "reason": "Tone or wording"})
    assert status_of(rid) == "rejected"
    cl.post("/rules", data={"pattern": "(unclosed", "reason": "x"})
    cl.post("/rules", data={"pattern": r"\bsynergy\b", "reason": "no buzzwords"})
    with engine.connect() as c:
        pats = c.execute(text("SELECT pattern FROM guardrail_rules")).scalars().all()
    assert r"\bsynergy\b" in pats and "(unclosed" not in pats
