"""Agent tools: read-only SQL, support tickets (DB or Zendesk), retention playbook RAG.
Every tool records what it returned in a RunContext so guardrails can verify citations."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import requests
import sqlglot
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from sqlalchemy import insert, text
from sqlglot import exp

from core import (AGENT_TABLES, FORBIDDEN_COLUMNS, MAX_SQL_ROWS, agent_engine, engine, mask_pii,
                  playbook_collection, query_log)


@dataclass
class RunContext:
    run_id: str
    account_id: str
    contact_names: list[str]
    query_refs: dict[str, str] = field(default_factory=dict)
    ticket_refs: set[str] = field(default_factory=set)
    playbook_refs: set[str] = field(default_factory=set)

    def evidence_ids(self) -> set[str]:
        return set(self.query_refs) | self.ticket_refs


def _log(ctx: RunContext, ref: str, kind: str, sql: str, rows: int) -> None:
    with engine.begin() as c:
        c.execute(insert(query_log).values(run_id=ctx.run_id, account_id=ctx.account_id, ref=ref, kind=kind,
                                           sql=sql, rows=rows, created_at=datetime.now()))


# ── read-only SQL ──────────────────────────────────────────────────────────────
_ACCOUNT_RE = re.compile(r"A-\d{5}")
_WRITE_NODES = tuple(getattr(exp, n) for n in ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable",
                                                "Merge", "Command", "TruncateTable", "Pragma", "Attach") if hasattr(exp, n))
_DIALECT = {"sqlite": "sqlite", "postgresql": "postgres"}.get(agent_engine.dialect.name)


def validate_sql(query: str, account_id: str, dialect: str | None = _DIALECT) -> str:
    try:
        stmts = [s for s in sqlglot.parse(query.strip().rstrip(";"), read=dialect) if s is not None]
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"could not parse SQL: {e}") from None
    if len(stmts) != 1:
        raise ValueError("send exactly one SELECT statement")
    tree = stmts[0]
    if not isinstance(tree, (exp.Select, exp.Union)) or tree.find(*_WRITE_NODES):
        raise ValueError("only SELECT queries are allowed")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
    if bad := tables - AGENT_TABLES:
        raise ValueError(f"tables not allowed: {sorted(bad)}; allowed: {sorted(AGENT_TABLES)}")
    if {c.name.lower() for c in tree.find_all(exp.Column)} & FORBIDDEN_COLUMNS:
        raise ValueError("contact columns are not accessible")
    if "accounts" in tables and any(not isinstance(s.parent, exp.Count) for s in tree.find_all(exp.Star)):
        raise ValueError("list columns explicitly when selecting from accounts")
    if set(_ACCOUNT_RE.findall(query)) != {account_id}:
        raise ValueError(f"filter on account_id = '{account_id}' and reference no other account")
    return tree.sql(dialect=dialect)


class SQLArgs(BaseModel):
    query: str = Field(..., description="One read-only SELECT over accounts, usage_daily or tickets, filtered to the current account_id.")


class ReadOnlySQLTool(BaseTool):
    name: str = "read_only_sql"
    description: str = (f"Run one read-only SQL SELECT on the telemetry warehouse (max {MAX_SQL_ROWS} rows). "
                        "Each successful call returns a query_ref (Q-xxxxxxxx) that you must cite as evidence.")
    args_schema: type[BaseModel] = SQLArgs
    _ctx: RunContext = PrivateAttr()

    def __init__(self, ctx: RunContext, **kw):
        super().__init__(**kw)
        self._ctx = ctx

    def _run(self, query: str) -> str:
        ctx = self._ctx
        try:
            sql = validate_sql(query, ctx.account_id)
            wrapped = f"SELECT * FROM ({sql}) AS q LIMIT {MAX_SQL_ROWS}".replace(":", "\\:")
            with agent_engine.connect() as c:
                df = pd.read_sql(text(wrapped), c)
        except Exception as e:  # returned to the agent so it can self-correct
            return f"ERROR: {e}"
        ref = "Q-" + hashlib.sha1(f"{ctx.run_id}|{sql}".encode()).hexdigest()[:8]
        ctx.query_refs[ref] = sql
        _log(ctx, ref, "sql", sql, len(df))
        return f"query_ref: {ref}\nrows: {len(df)}\n" + mask_pii(df.to_csv(index=False), ctx.contact_names)


# ── support tickets ────────────────────────────────────────────────────────────
_TICKET_COLS = ["ticket_id", "created_at", "subject", "body", "category", "priority", "status", "csat"]
ZENDESK = os.getenv("ZENDESK_SUBDOMAIN")


def _zendesk_tickets(account_id: str, since: datetime, limit: int) -> pd.DataFrame:
    base = f"https://{ZENDESK}.zendesk.com/api/v2"
    auth = (f"{os.environ['ZENDESK_EMAIL']}/token", os.environ["ZENDESK_API_TOKEN"])
    orgs = requests.get(f"{base}/organizations/search.json", params={"external_id": account_id},
                        auth=auth, timeout=20).json().get("organizations", [])
    if not orgs:
        return pd.DataFrame(columns=_TICKET_COLS)
    r = requests.get(f"{base}/organizations/{orgs[0]['id']}/tickets.json", params={"per_page": 100},
                     auth=auth, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame([dict(
        ticket_id=f"T-{t['id']}", created_at=pd.Timestamp(t["created_at"]).tz_localize(None),
        subject=t.get("subject") or "", body=t.get("description") or "", category=t.get("type") or "question",
        priority=t.get("priority") or "normal", status=t["status"],
        csat={"good": 5, "bad": 1}.get((t.get("satisfaction_rating") or {}).get("score")))
        for t in r.json().get("tickets", [])], columns=_TICKET_COLS)
    return df[df.created_at >= since].sort_values("created_at", ascending=False).head(limit)


def fetch_tickets(account_id: str, days: int, limit: int) -> pd.DataFrame:
    since = datetime.now() - timedelta(days=days)
    if ZENDESK:
        return _zendesk_tickets(account_id, since, limit)
    with engine.connect() as c:
        return pd.read_sql(text(f"SELECT {', '.join(_TICKET_COLS)} FROM tickets WHERE account_id = :a "
                                "AND created_at >= :s ORDER BY created_at DESC LIMIT :n"),
                           c, params={"a": account_id, "s": since, "n": limit}, parse_dates=["created_at"])


class TicketArgs(BaseModel):
    days: int = Field(60, description="Look-back window in days")
    limit: int = Field(10, description="Maximum tickets to return (<= 25)")


class RecentTicketsTool(BaseTool):
    name: str = "recent_support_tickets"
    description: str = "Fetch the current account's recent support tickets (PII masked). Cite ticket IDs (T-...) as evidence."
    args_schema: type[BaseModel] = TicketArgs
    _ctx: RunContext = PrivateAttr()

    def __init__(self, ctx: RunContext, **kw):
        super().__init__(**kw)
        self._ctx = ctx

    def _run(self, days: int = 60, limit: int = 10) -> str:
        ctx = self._ctx
        try:
            df = fetch_tickets(ctx.account_id, int(days), max(1, min(int(limit), 25)))
        except Exception as e:
            return f"ERROR: {e}"
        if df.empty:
            return f"No tickets in the last {days} days."
        ctx.ticket_refs |= set(df.ticket_id)
        _log(ctx, ",".join(df.ticket_id), "tickets", f"days={days} limit={limit}", len(df))
        lines = [f"{r.ticket_id} | {str(r.created_at)[:10]} | {r.category} | {r.priority} | {r.status} | "
                 f"csat={r.csat} | {r.subject} | {str(r.body).replace(chr(10), ' ')[:400]}" for r in df.itertuples()]
        return mask_pii("\n".join(lines), ctx.contact_names)


# ── playbook RAG ───────────────────────────────────────────────────────────────
class PlaybookArgs(BaseModel):
    query: str = Field(..., description="Situation needing guidance, e.g. 'unresolved export bug, frustrated finance team'")
    k: int = Field(4, description="Number of entries (<= 8)")


class PlaybookSearchTool(BaseTool):
    name: str = "retention_playbook_search"
    description: str = ("Semantic search over the approved Retention Playbook and past successful saves. "
                        "Only cite entry IDs returned here. 'permits' lists what an entry allows you to offer.")
    args_schema: type[BaseModel] = PlaybookArgs
    _ctx: RunContext = PrivateAttr()

    def __init__(self, ctx: RunContext, **kw):
        super().__init__(**kw)
        self._ctx = ctx

    def _run(self, query: str, k: int = 4) -> str:
        col = playbook_collection()
        res = col.query(query_texts=[query], n_results=max(1, min(int(k), 8, col.count())))
        ids, docs, metas = res["ids"][0], res["documents"][0], res["metadatas"][0]
        self._ctx.playbook_refs |= set(ids)
        return "\n\n".join(f"[{i}] permits: {m.get('permits') or 'none'}\n{d}" for i, d, m in zip(ids, docs, metas))
