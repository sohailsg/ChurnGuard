"""Deterministic guardrails: grounding, unapproved promises, PII / cross-customer leakage, placeholders."""
from __future__ import annotations

import re

from sqlalchemy import select

from core import EMAIL_RE, PHONE_RE, PLACEHOLDERS, accounts, engine, guardrail_rules, playbook

PROMISES = {
    "discount": re.compile(r"\b(discounts?|\d+\s?%\s?off|free (?:months?|seats?|upgrade)|refunds?|credits?|waived?|price (?:cut|reduction))\b", re.I),
    "roadmap": re.compile(r"\b(roadmap|next release|upcoming release|coming soon|in the next (?:sprint|quarter|version)|"
                          r"we will (?:build|ship|release|add|launch)|will be (?:released|available|fixed) (?:in|by|on))\b", re.I),
    "guarantee": re.compile(r"\bguarantee", re.I),
}
_ACCOUNT_RE = re.compile(r"\bA-\d{5}\b")
_INTERNAL_ID_RE = re.compile(r"\b(?:Q-[0-9a-f]{8}|T-\d+|PB-\w+)\b")


def _account_names() -> dict[str, str]:
    with engine.connect() as c:
        return dict(c.execute(select(accounts.c.account_id, accounts.c.name)).all())


def _rules() -> list[tuple[str, str]]:
    with engine.connect() as c:
        return c.execute(select(guardrail_rules.c.pattern, guardrail_rules.c.reason)
                         .where(guardrail_rules.c.active.is_(True))).all()


def permits_for(refs) -> set[str]:
    if not refs:
        return set()
    with engine.connect() as c:
        rows = c.execute(select(playbook.c.permits).where(playbook.c.entry_id.in_(list(refs)))).scalars().all()
    return {p.strip() for r in rows if r for p in r.split(",") if p.strip()}


def leakage_issues(text: str, account_id: str, contact_names: list[str]) -> list[str]:
    errs, low = [], text.lower()
    if EMAIL_RE.search(text) or PHONE_RE.search(text):
        errs.append("contains an email address or phone number")
    if others := set(_ACCOUNT_RE.findall(text)) - {account_id}:
        errs.append(f"mentions other account IDs {sorted(others)}")
    for n in contact_names:
        if any(re.search(rf"\b{re.escape(p)}\b", text, re.I) for p in (n, n.split()[0]) if len(p) > 2):
            errs.append("contains the contact's real name; use [CONTACT_FIRST_NAME]")
    names = _account_names()
    own = names.get(account_id, "").lower()
    if leaked := [v for k, v in names.items() if k != account_id and v.lower() in low and v.lower() not in own]:
        errs.append(f"mentions other customers {leaked[:3]}")
    return errs


def draft_issues(subject: str, body: str, account_id: str, contact_names: list[str], permits: set[str]) -> list[str]:
    text, errs = f"{subject}\n{body}", []
    for kind, rx in PROMISES.items():
        if kind not in permits and (m := rx.search(text)):
            errs.append(f"unapproved {kind} language: '{m.group(0)}'")
    for pattern, reason in _rules():
        try:
            if re.search(pattern, text, re.I):
                errs.append(f"rule violation: {reason}")
        except re.error:
            pass
    if bad := set(re.findall(r"\[[A-Z_]+\]", text)) - PLACEHOLDERS:
        errs.append(f"unknown placeholders {sorted(bad)}; use only {sorted(PLACEHOLDERS)}")
    if _INTERNAL_ID_RE.search(text):
        errs.append("draft exposes internal IDs (query refs, ticket or playbook IDs)")
    if not 50 <= len(body.split()) <= 260:
        errs.append(f"body is {len(body.split())} words; keep it between 50 and 260")
    return errs + leakage_issues(text, account_id, contact_names)


def check_profile(p, ctx) -> list[str]:
    errs = [] if p.account_id == ctx.account_id else [f"account_id must be {ctx.account_id}"]
    known = ctx.evidence_ids()
    for i, s in enumerate(p.signals):
        if missing := [e for e in s.evidence if e not in known]:
            errs.append(f"signal {i + 1} cites {missing}, which your tools never returned; cite only returned query_refs/ticket IDs")
    text = " ".join([p.likely_root_cause, *(s.detail for s in p.signals)])
    return errs + [e for e in leakage_issues(text, ctx.account_id, []) if "other" in e]


def check_intervention(iv, ctx) -> list[str]:
    errs = [] if iv.account_id == ctx.account_id else [f"account_id must be {ctx.account_id}"]
    refs = set(iv.playbook_refs)
    if bad := refs - ctx.playbook_refs:
        errs.append(f"playbook_refs {sorted(bad)} were not returned by retention_playbook_search")
    return errs + draft_issues(iv.draft_email.subject, iv.draft_email.body, ctx.account_id, ctx.contact_names,
                               permits_for(refs & ctx.playbook_refs))


def _strip_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()


def as_task_guardrail(check, ctx, model):
    """Adapter for CrewAI Task(guardrail=...): failing output is sent back to the agent with the reasons."""
    def _guard(output):
        obj = output.pydantic
        if obj is None:
            try:
                obj = model.model_validate_json(_strip_fences(output.raw))
            except Exception as e:
                return False, f"Output is not a valid {model.__name__} JSON object: {e}"
        errs = check(obj, ctx)
        return (False, "Revise your answer. Problems: " + "; ".join(errs)) if errs else (True, output)
    return _guard
