"""ChurnGuard core: settings, schema, Power BI views, PII masking, playbook index, synthetic data."""
from __future__ import annotations

import os
import re
from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import (Boolean, Column, Date, DateTime, Float, Index, Integer, MetaData, String, Table, Text,
                        create_engine, insert, select, text)

# ── settings ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///churnguard.db")
_ro_default = (f"sqlite:///file:{DATABASE_URL.removeprefix('sqlite:///')}?mode=ro&uri=true"
               if DATABASE_URL.startswith("sqlite:///") else DATABASE_URL)
# Postgres: point this at a role with SELECT-only grants on accounts, usage_daily, tickets.
AGENT_DATABASE_URL = os.getenv("AGENT_DATABASE_URL", _ro_default)

ANALYST_MODEL = os.getenv("ANALYST_MODEL", "anthropic/claude-sonnet-5-5")
CSM_MODEL = os.getenv("CSM_MODEL", "anthropic/claude-sonnet-5-5")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")  # e.g. http://localhost:11434 with ANALYST_MODEL=ollama/qwen2.5:7b
CREW_VERBOSE =os.getenv("CREW_VERBOSE", "0") == "1"

RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.5"))
MAX_DAILY_FLAGS = int(os.getenv("MAX_DAILY_FLAGS", "15"))
HOLDOUT_RATE = float(os.getenv("HOLDOUT_RATE", "0.2"))
COOLDOWN_DAYS = 14
HORIZON_DAYS = 30
MAX_SQL_ROWS = 200
MODEL_PATH = os.getenv("MODEL_PATH", "churn_model.joblib")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")

FEATURES = ("dashboards", "reports", "exports", "integrations", "alerts")
CORE_FEATURES = ("reports", "dashboards")
AGENT_TABLES = {"accounts", "usage_daily", "tickets"}
FORBIDDEN_COLUMNS = {"contact_name", "contact_email"}
PLACEHOLDERS = {"[CONTACT_FIRST_NAME]", "[CSM_NAME]"}

engine = create_engine(DATABASE_URL)
agent_engine = create_engine(AGENT_DATABASE_URL)

# ── schema ─────────────────────────────────────────────────────────────────────
md = MetaData()

accounts = Table(
    "accounts", md,
    Column("account_id", String, primary_key=True), Column("name", String), Column("plan", String),
    Column("arr", Float), Column("seats", Integer), Column("csm_owner", String), Column("signup_date", Date),
    Column("contact_name", String), Column("contact_email", String),
    Column("status", String),  # active | churned | downgraded
    Column("churn_date", Date))

usage_daily = Table(
    "usage_daily", md,
    Column("account_id", String), Column("activity_date", Date), Column("feature", String),
    Column("sessions", Integer), Column("active_users", Integer),
    Index("ix_usage_account_date", "account_id", "activity_date"))

tickets = Table(
    "tickets", md,
    Column("ticket_id", String, primary_key=True), Column("account_id", String, index=True),
    Column("created_at", DateTime), Column("subject", String), Column("body", Text), Column("category", String),
    Column("priority", String), Column("status", String), Column("csat", Integer))

playbook = Table(
    "playbook", md,
    Column("entry_id", String, primary_key=True), Column("title", String), Column("trigger", Text),
    Column("actions", Text), Column("constraints", Text),
    Column("permits", String, default=""),  # comma list, e.g. "discount"
    Column("source", String, default="seed"),  # seed | learned
    Column("created_at", DateTime))

interventions = Table(
    "churn_interventions", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String), Column("run_date", Date), Column("account_id", String, index=True),
    Column("risk_score", Float), Column("risk_tier", String), Column("arr", Float),
    Column("profile_json", Text), Column("intervention_json", Text),
    Column("guardrail_passed", Boolean), Column("guardrail_issues", Text),
    # pending_review | approved | edited | rejected | sent | holdout | guardrail_failed | error
    Column("status", String, index=True),
    Column("final_subject", Text), Column("final_body", Text), Column("edit_similarity", Float),
    Column("rejection_reason", Text), Column("reviewer", String), Column("reviewed_at", DateTime),
    Column("sent_at", DateTime), Column("outcome", String), Column("outcome_at", Date), Column("error", Text))

query_log = Table(
    "agent_query_log", md,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("run_id", String),
    Column("account_id", String), Column("ref", Text), Column("kind", String), Column("sql", Text),
    Column("rows", Integer), Column("created_at", DateTime))

guardrail_rules = Table(
    "guardrail_rules", md,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("pattern", Text),
    Column("reason", Text), Column("active", Boolean, default=True), Column("created_at", DateTime))

AGENT_SCHEMA_HINT = """\
accounts(account_id, name, plan, arr, seats, csm_owner, signup_date, status, churn_date)
usage_daily(account_id, activity_date, feature, sessions, active_users)  -- one row per account/day/feature; features: dashboards, reports, exports, integrations, alerts
tickets(ticket_id, account_id, created_at, subject, body, category, priority, status, csat)  -- category: bug, how_to, billing, account, feature_request"""

# ── Power BI views ─────────────────────────────────────────────────────────────
_ok = "status IN ('approved','edited','sent')"
VIEWS = {
    "v_review_queue": """
        SELECT i.id, i.run_date, i.account_id, a.name AS account_name, a.plan, a.csm_owner, i.arr,
               i.risk_score, i.risk_tier, i.risk_score * i.arr AS priority, i.status, i.guardrail_issues,
               i.profile_json, i.intervention_json
        FROM churn_interventions i JOIN accounts a ON a.account_id = i.account_id
        WHERE i.status IN ('pending_review','guardrail_failed')""",
    "v_kpi_daily": f"""
        SELECT run_date,
               COUNT(*) AS flagged,
               SUM(CASE WHEN status = 'holdout' THEN 1 ELSE 0 END) AS holdout,
               SUM(CASE WHEN {_ok} THEN 1 ELSE 0 END) AS approved,
               SUM(CASE WHEN status = 'edited' THEN 1 ELSE 0 END) AS edited,
               SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected,
               SUM(CASE WHEN status = 'pending_review' THEN 1 ELSE 0 END) AS pending,
               SUM(CASE WHEN status IN ('guardrail_failed','error') THEN 1 ELSE 0 END) AS failed,
               1.0 * SUM(CASE WHEN {_ok} THEN 1 ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN {_ok} OR status = 'rejected' THEN 1 ELSE 0 END), 0) AS approval_rate,
               AVG(edit_similarity) AS avg_edit_similarity
        FROM churn_interventions GROUP BY run_date""",
    "v_outcomes": f"""
        SELECT CASE WHEN {_ok} THEN 'intervened' WHEN status = 'holdout' THEN 'holdout'
                    ELSE 'not_actioned' END AS arm,
               risk_tier,
               COUNT(*) AS accounts,
               SUM(CASE WHEN outcome = 'retained' THEN 1 ELSE 0 END) AS retained,
               SUM(CASE WHEN outcome = 'churned' THEN 1 ELSE 0 END) AS churned,
               1.0 * SUM(CASE WHEN outcome = 'retained' THEN 1 ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END), 0) AS save_rate,
               SUM(CASE WHEN outcome = 'retained' THEN arr ELSE 0 END) AS arr_retained
        FROM churn_interventions WHERE status NOT IN ('error','guardrail_failed')
        GROUP BY 1, 2""",
}


def init_db(reset: bool = False) -> None:
    with engine.begin() as c:
        for v in VIEWS:
            c.execute(text(f"DROP VIEW IF EXISTS {v}"))
    if reset:
        md.drop_all(engine)
    md.create_all(engine)
    with engine.begin() as c:
        for v, sql in VIEWS.items():
            c.execute(text(f"CREATE VIEW {v} AS {sql}"))


# ── PII masking ────────────────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}(?!\w)")


def mask_pii(s: str, names: list[str] = ()) -> str:
    s = PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", s))
    for n in names:
        for part in [n, *n.split()]:
            if len(part) > 2:
                s = re.sub(rf"\b{re.escape(part)}\b", "[CONTACT]", s, flags=re.I)
    return s


def contact_names(account_id: str) -> list[str]:
    with engine.connect() as c:
        return [n for n in c.execute(select(accounts.c.contact_name)
                                     .where(accounts.c.account_id == account_id)).scalars() if n]


# ── retention playbook ─────────────────────────────────────────────────────────
SEED_PLAYBOOK = [
    dict(entry_id="PB-002", title="Value recap", permits="",
         trigger="Gradual usage decline without complaints; low engagement; renewal approaching.",
         actions="Send a short recap of what the team achieved with the product and propose a 20-minute call to re-align on goals.",
         constraints="No pricing discussion in the first touch."),
    dict(entry_id="PB-005", title="Re-onboarding and guided training", permits="",
         trigger="How-to tickets, low feature breadth, new users struggling, onboarding gap.",
         actions="Offer a live 30-minute training on the exact features they struggle with, share the matching help guides, book a follow-up in two weeks.",
         constraints="Keep it about their workflow, not a product tour."),
    dict(entry_id="PB-009", title="Champion change", permits="",
         trigger="Admin or main champion left; active users dropping; admin access tickets.",
         actions="Restore admin access fast, ask who now owns the tool internally, offer a handover session for the new owner.",
         constraints="Do not speculate about the departed person."),
    dict(entry_id="PB-014", title="Technical escalation for unresolved bugs", permits="",
         trigger="Repeated bug tickets, failures after a release, frustrated sentiment.",
         actions="Acknowledge the specific issue and its business impact, confirm it is escalated to engineering with a named owner, offer any workaround, commit to a status update on a specific day.",
         constraints="Never promise a fix date or release version; commit only to status updates."),
    dict(entry_id="PB-018", title="Feature request acknowledgment", permits="",
         trigger="Customer needs a capability the product does not have.",
         actions="Thank them, explain what is possible today, log the request with product, offer a solutions-engineer call.",
         constraints="Never commit to the roadmap or timelines."),
    dict(entry_id="PB-021", title="Commercial value review", permits="discount",
         trigger="Pricing objections, seat-reduction questions, renewal cost pushback.",
         actions="Offer a value review using their usage data. The account owner may offer seat right-sizing or a renewal discount of up to 15%, subject to manager approval.",
         constraints="Only for a confirmed pricing objection; max 15%; always framed as subject to approval."),
    dict(entry_id="PB-027", title="Integration workaround", permits="",
         trigger="Missing integration, webhook failures, manual data imports.",
         actions="Offer a solutions-engineering session to set up an API or webhook workaround and share the relevant docs.",
         constraints="Do not promise native integrations."),
    dict(entry_id="PB-031", title="Executive check-in", permits="",
         trigger="High-ARR or Enterprise account showing high risk.",
         actions="Propose a short call between the customer's sponsor and our VP of Customer Success; the CSM stays the primary contact.",
         constraints="Enterprise plan or ARR above 50k only."),
    dict(entry_id="PB-036", title="Re-engagement for dormant accounts", permits="",
         trigger="No activity for 10+ days, near-zero sessions.",
         actions="Low-pressure note asking whether priorities changed, offer to help reset workflows, share one quick win.",
         constraints="No guilt-tripping or urgency tactics."),
    dict(entry_id="PB-040", title="Service recovery after poor support", permits="",
         trigger="Low CSAT, open high-priority tickets.",
         actions="Apologise for the specific experience, assign a single point of contact, offer a call to walk through open issues.",
         constraints="Do not blame the support team or the customer."),
]


def seed_playbook() -> None:
    with engine.begin() as c:
        have = set(c.execute(select(playbook.c.entry_id)).scalars())
        rows = [dict(e, source="seed", created_at=datetime.now()) for e in SEED_PLAYBOOK if e["entry_id"] not in have]
        if rows:
            c.execute(insert(playbook), rows)


def playbook_collection():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_PATH).get_or_create_collection("playbook")


def index_playbook() -> None:
    with engine.connect() as c:
        df = pd.read_sql(select(playbook), c)
    if df.empty:
        return
    playbook_collection().upsert(
        ids=df.entry_id.tolist(),
        documents=[f"{r.title}\nTrigger: {r.trigger}\nActions: {r.actions}\nConstraints: {r.constraints or 'none'}"
                   for r in df.itertuples()],
        metadatas=[{"title": r.title, "permits": r.permits or "", "source": r.source} for r in df.itertuples()])


def bootstrap() -> None:
    init_db()
    seed_playbook()
    index_playbook()


# ── synthetic data (for backtest + demo) ───────────────────────────────────────
_FIRST = ["Priya", "Rahul", "Emma", "Liam", "Aisha", "Kenji", "Sofia", "Noah", "Meera", "Arjun", "Olivia", "Daniel"]
_LAST = ["Sharma", "Patel", "Smith", "Chen", "Khan", "Garcia", "Iyer", "Brown", "Nair", "Wilson"]
_PREFIX = ["Nimbus", "Vertex", "Bluepeak", "Orbit", "Kestrel", "Lumen", "Cobalt", "Harbor", "Summit", "Atlas", "Quartz", "Pioneer"]
_SUFFIX = ["Logistics", "Health", "Retail", "Labs", "Finance", "Foods", "Media", "Energy", "Systems", "Travel"]
_CSMS = ["Ananya Rao", "Mark Ellis", "Fatima Noor"]
_CAUSE_TICKETS = {
    "export_bug": [("CSV export fails since v4.2", "Our scheduled CSV exports have failed three times this week since the v4.2 update. Finance depends on these.", "bug", "high"),
                   ("Export times out on large reports", "Exports over 10k rows time out and we are rebuilding them by hand.", "bug", "high")],
    "onboarding_gap": [("How do we share team dashboards?", "New team members can't work out how to share dashboards. Is there a guide?", "how_to", "normal"),
                       ("Confused by report builder filters", "We don't understand how filters combine in the report builder.", "how_to", "normal")],
    "champion_left": [("Change of account admin", "Our admin has left the company. Who do we contact to transfer ownership?", "account", "normal"),
                      ("No one has admin rights", "Since our admin left nobody can manage users or settings.", "account", "high")],
    "pricing": [("Renewal quote much higher than expected", "We need to justify the renewal cost internally and it's hard right now.", "billing", "normal"),
                ("How do we reduce seats?", "We'd like to understand how to reduce seats before renewal.", "billing", "normal")],
    "missing_integration": [("Need a Salesforce sync", "Manual CSV imports into Salesforce are not sustainable for us.", "feature_request", "normal"),
                            ("Webhook to CRM drops events", "The webhook to our CRM drops events every day.", "bug", "high")],
}
_BASE_TICKETS = [("How to schedule a report", "Can reports be scheduled weekly?", "how_to", "low"),
                 ("Invoice copy", "Please send a copy of last month's invoice.", "billing", "low"),
                 ("Dark mode request", "Any chance of a dark mode?", "feature_request", "low"),
                 ("Dashboard loads slowly", "One dashboard takes ~10s to load.", "bug", "normal")]


def seed_synthetic(n_accounts: int = 200, days: int = 180, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(date.today())
    dates = pd.date_range(today - pd.Timedelta(days=days - 1), today, freq="D")
    accts, usage, tix, tid = [], [], [], 100000

    for i in range(n_accounts):
        aid = f"A-{10000 + i}"
        plan = str(rng.choice(["Starter", "Growth", "Enterprise"], p=[0.4, 0.4, 0.2]))
        lo, hi, price = {"Starter": (3, 15, 300), "Growth": (15, 60, 600), "Enterprise": (60, 250, 1100)}[plan]
        seats = int(rng.integers(lo, hi))
        name = f"{_PREFIX[i % 12]} {_SUFFIX[(i // 12) % 10]}" + (" Group" if i >= 120 else "")
        first, last = str(rng.choice(_FIRST)), str(rng.choice(_LAST))
        contact, email = f"{first} {last}", f"{first.lower()}.{last.lower()}@{name.lower().replace(' ', '')}.com"

        churns = rng.random() < 0.28
        cause = str(rng.choice(list(_CAUSE_TICKETS))) if churns else None
        churn_idx = int(rng.integers(50, days + 30)) if churns else days + 999  # may be in the future
        decline = int(rng.integers(18, 45)) if churns else 1
        d0 = churn_idx - decline if churns else days + 999
        downgrade = churns and rng.random() < 0.25
        dip = int(rng.integers(30, days - 20)) if (not churns and rng.random() < 0.15) else None
        base = {f: 0.0 if (f in ("integrations", "alerts") and rng.random() < 0.3) else rng.uniform(0.3, 1.2) * seats
                for f in FEATURES}
        top, users_base = max(base.values()), seats * rng.uniform(0.5, 0.9)

        for d, day in enumerate(dates):
            if d >= churn_idx and not downgrade:
                break
            m = 1.0
            if d >= d0:
                m = 0.35 if d >= churn_idx else max(0.05, 1 - 0.9 * (d - d0) / decline)
            if dip is not None and dip <= d < dip + 12:
                m = 0.55
            wk = 0.3 if day.weekday() >= 5 else 1.0
            au_m = m * (0.6 if cause == "champion_left" and d >= d0 else 1.0)
            for f, b in base.items():
                if b == 0:
                    continue
                fm = m * (0.3 if d >= d0 and ((cause == "export_bug" and f == "exports")
                                              or (cause == "missing_integration" and f == "integrations")) else 1)
                usage.append((aid, day.date(), f, int(rng.poisson(b * fm * wk)),
                              int(min(seats, rng.poisson(users_base * au_m * wk * b / top)))))

            in_decline = d0 <= d < churn_idx
            if rng.random() < (0.12 if in_decline else 0.025):
                pool = _CAUSE_TICKETS[cause] if in_decline and rng.random() < 0.8 else _BASE_TICKETS
                subj, body, cat, pri = pool[int(rng.integers(len(pool)))]
                status = "open" if in_decline and d > days - 10 else "solved"
                tix.append(dict(
                    ticket_id=f"T-{tid}", account_id=aid, created_at=day + pd.Timedelta(hours=int(rng.integers(8, 19))),
                    subject=subj, body=f"{body}\n\nThanks,\n{contact} | {email} | +1 415 555 {int(rng.integers(1000, 9999))}",
                    category=cat, priority=pri, status=status,
                    csat=None if status == "open" else int(rng.integers(1, 4) if in_decline else rng.integers(4, 6))))
                tid += 1

        past = churns and churn_idx < days
        accts.append(dict(
            account_id=aid, name=name, plan=plan, arr=round(seats * price * rng.uniform(0.9, 1.1), 2), seats=seats,
            csm_owner=str(rng.choice(_CSMS)),
            signup_date=(dates[0] - pd.Timedelta(days=int(rng.integers(60, 900)))).date(),
            contact_name=contact, contact_email=email,
            status=("downgraded" if downgrade else "churned") if past else "active",
            churn_date=dates[churn_idx].date() if past else None))

    init_db(reset=True)
    pd.DataFrame(accts).to_sql("accounts", engine, if_exists="append", index=False)
    pd.DataFrame(usage, columns=["account_id", "activity_date", "feature", "sessions", "active_users"]).to_sql(
        "usage_daily", engine, if_exists="append", index=False, chunksize=20000)
    pd.DataFrame(tix).to_sql("tickets", engine, if_exists="append", index=False)
    seed_playbook()
    index_playbook()
    print(f"Seeded {len(accts)} accounts, {len(usage):,} usage rows, {len(tix)} tickets.")
