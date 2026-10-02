import json
from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import text, update

import core
import pipeline
from core import engine, interventions
from schemas import DraftEmail, Intervention, RiskProfile, Signal

BODY = ("Hi [CONTACT_FIRST_NAME],\n\nI noticed your team has been having trouble with scheduled exports lately and wanted to "
        "reach out personally. I have raised it with our engineering team and will send a status update on Friday. Meanwhile I can "
        "show you a manual workaround so your finance team is not blocked. Would a short call this week suit you? I would also "
        "like to hear whether anything else has been getting in the way of your reports.\n\nBest,\n[CSM_NAME]")


def fake_run_account(ctx, inputs):
    ctx.query_refs["Q-aaaaaaaa"], ctx.playbook_refs = "sql", {"PB-014"}
    p = RiskProfile(account_id=ctx.account_id, risk_score=0.9, risk_tier="high", sentiment="frustrated",
                    likely_root_cause="export bug", confidence="high",
                    signals=[Signal(type="usage", detail="exports down 60%", evidence=["Q-aaaaaaaa"])])
    iv = Intervention(account_id=ctx.account_id, strategy="Technical escalation", rationale="r", playbook_refs=["PB-014"],
                      draft_email=DraftEmail(subject="Your exports", body=BODY))
    return p, iv


@pytest.fixture()
def run(seeded, monkeypatch):
    import crew
    with engine.begin() as c:
        c.execute(text("DELETE FROM churn_interventions"))
    pipeline.train()
    monkeypatch.setattr(crew, "run_account", fake_run_account)
    return pipeline.run_daily


def rows():
    return pd.read_sql(text("SELECT * FROM churn_interventions"), engine)


def test_features_no_lookahead_and_drop(seeded):
    acc, use, tix = pipeline.load_frames()
    as_of = use.activity_date.max() - pd.Timedelta(days=40)
    f_all = pipeline.compute_features(acc, use, tix, as_of)
    f_cut = pipeline.compute_features(acc, use[use.activity_date <= as_of], tix[tix.created_at <= as_of + pd.Timedelta(days=1)], as_of)
    pd.testing.assert_frame_equal(f_all, f_cut)  # future data must not change features


def test_core_drop_known_value():
    as_of = pd.Timestamp("2026-03-01")
    acc = pd.DataFrame([dict(account_id="A-1", name="x", plan="Growth", arr=1000.0, seats=10, csm_owner="c",
                             signup_date=pd.Timestamp("2025-01-01"), status="active", churn_date=pd.NaT)])
    days = pd.date_range(as_of - pd.Timedelta(days=27), as_of)
    use = pd.DataFrame([dict(account_id="A-1", activity_date=d, feature="reports",
                             sessions=10 if d <= as_of - pd.Timedelta(days=14) else 5, active_users=5) for d in days])
    tix = pd.DataFrame(columns=["ticket_id", "account_id", "created_at", "category", "priority", "csat"])
    f = pipeline.compute_features(acc, use, tix, as_of)
    assert f.loc["A-1", "core_drop"] == pytest.approx(0.5)


def test_training_labels_match_churn_dates(seeded):
    acc, use, tix = pipeline.load_frames()
    ds = pipeline.training_set(acc, use, tix)
    churn = acc.set_index("account_id").churn_date
    pos = ds[ds.label == 1]
    assert len(pos) > 0
    cd = churn.reindex(pos.account_id).values
    assert ((cd > pos.as_of.values) & (cd <= pos.as_of.values + pd.Timedelta(days=core.HORIZON_DAYS))).all()


def test_run_daily_stages_rows(run, monkeypatch):
    monkeypatch.setattr(core, "HOLDOUT_RATE", 0.5)
    run(limit=8)
    df = rows()
    assert len(df) > 0
    assert set(df.status) <= {"pending_review", "holdout", "guardrail_failed", "error"}
    pend = df[df.status == "pending_review"]
    assert (pend.guardrail_passed == True).all()  # noqa: E712
    assert pend.final_body.str.contains(r"\[CONTACT_FIRST_NAME\]").all()
    assert df[df.status == "holdout"].intervention_json.isna().all()
    n = len(df)
    run(limit=8)  # cooldown: already-flagged accounts are not flagged again
    assert set(rows().account_id.value_counts()) == {1} and len(rows()) >= n


def test_agent_exception_recorded_as_error(run, monkeypatch):
    import crew
    monkeypatch.setattr(core, "HOLDOUT_RATE", 0.0)
    monkeypatch.setattr(crew, "run_account", lambda c, i: (_ for _ in ()).throw(RuntimeError("llm down")))
    run(limit=2)
    df = rows()
    assert len(df) > 0 and set(df.status) == {"error"} and df.error.str.contains("llm down").all()


def test_guardrail_failure_is_staged(run, monkeypatch):
    import crew
    monkeypatch.setattr(core, "HOLDOUT_RATE", 0.0)

    def leaky(ctx, inputs):
        p, iv = fake_run_account(ctx, inputs)
        return p, iv.model_copy(update={"draft_email": DraftEmail(subject="s", body=BODY + "\nwe guarantee a fix")})
    monkeypatch.setattr(crew, "run_account", leaky)
    run(limit=2)
    df = rows()
    assert len(df) > 0 and set(df.status) == {"guardrail_failed"} and df.guardrail_issues.str.contains("guarantee").all()


def test_outcomes_and_learn(run, monkeypatch):
    monkeypatch.setattr(core, "HOLDOUT_RATE", 0.0)
    run(limit=3)
    old = date.today() - timedelta(days=40)
    with engine.begin() as c:
        c.execute(update(interventions).values(run_date=old, status="approved", final_subject="Your exports"))
        c.execute(text("UPDATE accounts SET churn_date = NULL"))
    pipeline.update_outcomes()
    assert set(rows().outcome) == {"retained"}
    before = pd.read_sql(text("SELECT COUNT(*) n FROM playbook"), engine).n[0]
    pipeline.learn()
    after = pd.read_sql(text("SELECT COUNT(*) n FROM playbook"), engine).n[0]
    assert after > before
    learned = pd.read_sql(text("SELECT * FROM playbook WHERE source='learned'"), engine)
    assert not learned.actions.str.contains("@").any()
    for v in core.VIEWS:  # Power BI views stay queryable
        pd.read_sql(text(f"SELECT * FROM {v}"), engine)
