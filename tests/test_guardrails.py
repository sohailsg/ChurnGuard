from datetime import datetime

import pytest
from sqlalchemy import insert

import guardrails
from core import engine, guardrail_rules
from guardrails import as_task_guardrail, check_intervention, check_profile, draft_issues
from schemas import DraftEmail, Intervention, RiskProfile, Signal
from tools import RunContext

GOOD = ("Hi [CONTACT_FIRST_NAME],\n\nI noticed your team has been running into trouble with scheduled exports over the past "
        "few weeks and I wanted to reach out personally. I have raised the issue with our engineering team and will send you a "
        "status update on Friday. In the meantime I can walk you through a manual workaround so your finance team is not blocked. "
        "Would a short call this week suit you? I would also like to hear whether anything else has been getting in the way of "
        "your reports.\n\nBest,\n[CSM_NAME]")


def issues(body=GOOD, subject="Your exports", permits=frozenset(), acct=("A-10000", ["Priya Sharma"])):
    return draft_issues(subject, body, acct[0], acct[1], set(permits))


def test_clean_draft_passes(seeded):
    assert issues() == []


@pytest.mark.parametrize("text,frag", [
    ("We can offer you a 15% off discount.", "discount"),
    ("This will be fixed in the next release.", "roadmap"),
    ("We guarantee this is resolved.", "guarantee"),
    ("A refund is possible.", "discount"),
])
def test_promises_blocked(seeded, text, frag):
    assert any(frag in i for i in issues(GOOD + "\n" + text))


def test_promise_allowed_when_permitted(seeded):
    assert not any("discount" in i for i in issues(GOOD + "\nA discount may be possible.", permits={"discount"}))


@pytest.mark.parametrize("text,frag", [
    ("Reach me at john@example.com", "email"),
    ("Call me on 415 555 1234", "email address or phone"),
    ("Thanks Priya for your help", "real name"),
    ("See account A-10005 for details", "other account"),
    ("ref Q-1234abcd", "internal IDs"),
    ("see T-100012", "internal IDs"),
    ("Hello [FOO_BAR]", "placeholders"),
])
def test_leakage_and_format(seeded, text, frag):
    assert any(frag in i for i in issues(GOOD + "\n" + text))


def test_other_customer_name_blocked(seeded):
    names = guardrails._account_names()
    other = next(v for k, v in names.items() if k != "A-10000" and v != names["A-10000"])
    assert any("other customers" in i for i in issues(GOOD + f"\nWe also help {other}."))


def test_length_limits(seeded):
    assert any("words" in i for i in issues("Hi [CONTACT_FIRST_NAME], short.\n[CSM_NAME]"))


def test_custom_rule(seeded):
    with engine.begin() as c:
        c.execute(insert(guardrail_rules).values(pattern=r"\bbusiness impact\b", reason="no jargon", active=True,
                                                 created_at=datetime.now()))
    assert any("no jargon" in i for i in issues(GOOD + "\nThe business impact is large."))


def ctx():
    c = RunContext(run_id="R-1", account_id="A-10000", contact_names=[])
    c.query_refs["Q-aaaaaaaa"], c.ticket_refs, c.playbook_refs = "sql", {"T-1"}, {"PB-014"}
    return c


def profile(evidence):
    return RiskProfile(account_id="A-10000", risk_score=0.8, risk_tier="high", sentiment="frustrated",
                       likely_root_cause="export bug", confidence="high",
                       signals=[Signal(type="usage", detail="down 40%", evidence=evidence)])


def test_profile_evidence_must_be_returned(seeded):
    assert check_profile(profile(["Q-aaaaaaaa", "T-1"]), ctx()) == []
    assert any("never returned" in e for e in check_profile(profile(["Q-deadbeef"]), ctx()))


def test_intervention_refs_must_be_returned(seeded):
    def iv(refs):
        return Intervention(account_id="A-10000", strategy="s", rationale="r", playbook_refs=refs,
                            draft_email=DraftEmail(subject="Your exports", body=GOOD))
    assert check_intervention(iv(["PB-014"]), ctx()) == []
    assert any("not returned" in e for e in check_intervention(iv(["PB-999"]), ctx()))


def test_task_guardrail_adapter(seeded):
    class Out:
        def __init__(self, pyd=None, raw=""):
            self.pydantic, self.raw = pyd, raw
    g = as_task_guardrail(check_profile, ctx(), RiskProfile)
    assert g(Out(profile(["Q-aaaaaaaa"])))[0] is True
    ok, msg = g(Out(profile(["Q-deadbeef"])))
    assert ok is False and "Revise" in msg
    assert g(Out(raw="not json"))[0] is False
    assert g(Out(raw="```json\n" + profile(["T-1"]).model_dump_json() + "\n```"))[0] is True
