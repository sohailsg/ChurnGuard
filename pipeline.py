"""ChurnGuard pipeline.
    python pipeline.py seed       # synthetic demo data + playbook (dev only)
    python pipeline.py init       # create tables/views + seed playbook on real data
    python pipeline.py backtest   # Phase 0: time-split evaluation vs simple rule
    python pipeline.py train      # fit risk model on all history
    python pipeline.py run        # daily: score -> rank -> holdout -> agents -> guardrails -> staging
    python pipeline.py outcomes   # label 30-day outcomes for flagged accounts
    python pipeline.py learn      # successful saves -> playbook; rejection reasons report
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sqlalchemy import insert, text, update

import core
from core import engine, interventions, playbook

FEATURE_COLS = ["core_drop", "total_drop", "au_drop", "seat_util", "breadth_change", "days_inactive",
                "tickets_21", "bug_tickets_21", "high_pri_21", "csat_60", "tenure_days", "log_arr"]
D = lambda n: pd.Timedelta(days=n)  # noqa: E731


# ── features ───────────────────────────────────────────────────────────────────
def load_frames(since: date | None = None):
    p = {"s": since} if since else {}
    with engine.connect() as c:
        acc = pd.read_sql(text("SELECT account_id, name, plan, arr, seats, csm_owner, signup_date, status, churn_date "
                               "FROM accounts"), c, parse_dates=["signup_date", "churn_date"])
        use = pd.read_sql(text("SELECT account_id, activity_date, feature, sessions, active_users FROM usage_daily"
                               + (" WHERE activity_date >= :s" if since else "")), c, params=p, parse_dates=["activity_date"])
        tix = pd.read_sql(text("SELECT ticket_id, account_id, created_at, category, priority, csat FROM tickets"
                               + (" WHERE created_at >= :s" if since else "")), c, params=p, parse_dates=["created_at"])
    return acc, use, tix


def compute_features(acc, use, tix, as_of) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of).normalize()
    live = acc[(acc.signup_date <= as_of - D(28)) & (acc.churn_date.isna() | (acc.churn_date > as_of))].set_index("account_id")
    f = pd.DataFrame(index=live.index)
    u = use[(use.activity_date > as_of - D(28)) & (use.activity_date <= as_of) & use.account_id.isin(f.index)]
    cur, prev = u[u.activity_date > as_of - D(14)], u[u.activity_date <= as_of - D(14)]

    al = lambda s: s.reindex(f.index).fillna(0)  # noqa: E731
    drop = lambda c, p: 1 - al(c) / al(p).clip(lower=1)  # noqa: E731
    sess = lambda df, feats=None: (df[df.feature.isin(feats)] if feats else df).groupby("account_id").sessions.sum()  # noqa: E731
    au = lambda df: df.groupby(["account_id", "activity_date"]).active_users.max().groupby("account_id").mean()  # noqa: E731
    breadth = lambda df: df[df.sessions > 0].groupby("account_id").feature.nunique()  # noqa: E731

    f["core_drop"] = drop(sess(cur, core.CORE_FEATURES), sess(prev, core.CORE_FEATURES))
    f["total_drop"] = drop(sess(cur), sess(prev))
    f["au_drop"] = drop(au(cur), au(prev))
    f["seat_util"] = al(au(cur)) / live.seats.clip(lower=1)
    f["breadth_change"] = al(breadth(cur)) - al(breadth(prev))
    last = use[(use.activity_date <= as_of) & (use.sessions > 0)].groupby("account_id").activity_date.max()
    f["days_inactive"] = (as_of - last.reindex(f.index)).dt.days.fillna(90).clip(upper=90)

    t = tix[tix.account_id.isin(f.index) & (tix.created_at < as_of + D(1))]
    t21 = t[t.created_at > as_of - D(21)]
    cnt = lambda df: al(df.groupby("account_id").size())  # noqa: E731
    f["tickets_21"] = cnt(t21)
    f["bug_tickets_21"] = cnt(t21[t21.category == "bug"])
    f["high_pri_21"] = cnt(t21[t21.priority.isin(["high", "urgent"])])
    f["csat_60"] = t[t.created_at > as_of - D(60)].groupby("account_id").csat.mean().reindex(f.index).fillna(4.5)
    f["tenure_days"] = (as_of - live.signup_date).dt.days
    f["log_arr"] = np.log1p(live.arr)
    return f.fillna(0)


def training_set(acc, use, tix, step: int = 7) -> pd.DataFrame:
    churn = acc.set_index("account_id").churn_date
    rows = []
    for as_of in pd.date_range(use.activity_date.min() + D(28), use.activity_date.max() - D(core.HORIZON_DAYS), freq=f"{step}D"):
        f = compute_features(acc, use, tix, as_of)
        cd = churn.reindex(f.index)
        f["label"] = ((cd > as_of) & (cd <= as_of + D(core.HORIZON_DAYS))).astype(int)
        f["as_of"] = as_of
        rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)


# ── model ──────────────────────────────────────────────────────────────────────
def make_model():
    return CalibratedClassifierCV(HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=250, class_weight="balanced", random_state=0), method="sigmoid", cv=3)


def train() -> None:
    ds = training_set(*load_frames())
    model = make_model().fit(ds[FEATURE_COLS], ds.label)
    joblib.dump({"model": model, "features": FEATURE_COLS, "trained_at": datetime.now().isoformat()}, core.MODEL_PATH)
    print(f"Trained on {len(ds):,} snapshots ({ds.label.sum()} positives) -> {core.MODEL_PATH}")


def _metrics(te: pd.DataFrame, score: str, flagged: pd.Series, k: int) -> list[float]:
    top = te.sort_values(score, ascending=False).groupby("as_of").head(k)
    auc = roc_auc_score(te.label, te[score]) if te.label.nunique() > 1 else float("nan")
    return [auc, precision_score(te.label, flagged, zero_division=0),
            recall_score(te.label, flagged, zero_division=0), top.groupby("as_of").label.mean().mean()]


def backtest(test_frac: float = 0.3) -> None:
    ds = training_set(*load_frames())
    dates = np.sort(ds.as_of.unique())
    cut = dates[int(len(dates) * (1 - test_frac))]
    tr, te = ds[ds.as_of < cut], ds[ds.as_of >= cut].copy()
    te["p"] = make_model().fit(tr[FEATURE_COLS], tr.label).predict_proba(te[FEATURE_COLS])[:, 1]
    k = core.MAX_DAILY_FLAGS
    report = pd.DataFrame({
        "model": _metrics(te, "p", te.p >= core.RISK_THRESHOLD, k),
        "rule: core usage drop >30%": _metrics(te, "core_drop", te.core_drop > 0.3, k),
    }, index=["ROC AUC", f"precision @ p>={core.RISK_THRESHOLD}", f"recall @ p>={core.RISK_THRESHOLD}", f"precision @ top {k}/day"])
    print(f"Train snapshots: {len(tr):,} | test: {len(te):,} ({te.label.mean():.1%} churn base rate) | cut: {pd.Timestamp(cut).date()}")
    print(report.round(3).to_string())


# ── daily run ──────────────────────────────────────────────────────────────────
def tier(p: float) -> str:
    return "high" if p >= 0.75 else "medium" if p >= core.RISK_THRESHOLD else "low"


def model_facts(r) -> str:
    return "\n".join([
        f"- core feature (reports, dashboards) sessions, last 14d vs prior 14d: {-r.core_drop:+.0%}",
        f"- all feature sessions: {-r.total_drop:+.0%}; daily active users: {-r.au_drop:+.0%}; seat utilisation {r.seat_util:.0%}",
        f"- features in use change: {r.breadth_change:+.0f}; days since last activity: {r.days_inactive:.0f}",
        f"- tickets last 21d: {r.tickets_21:.0f} (bugs {r.bug_tickets_21:.0f}, high priority {r.high_pri_21:.0f}); avg CSAT 60d: {r.csat_60:.1f}"])


def _write(row: dict) -> None:
    with engine.begin() as c:
        c.execute(insert(interventions).values(**row))


def run_daily(limit: int | None = None) -> None:
    from crew import run_account
    from guardrails import check_intervention, check_profile
    from tools import RunContext

    bundle = joblib.load(core.MODEL_PATH)
    acc, use, tix = load_frames(since=date.today() - timedelta(days=95))
    f = compute_features(acc, use, tix, date.today())
    f["risk_score"] = bundle["model"].predict_proba(f[bundle["features"]])[:, 1]
    f = f.join(acc.set_index("account_id")[["name", "plan", "seats", "arr", "csm_owner"]].rename(columns={"name": "company"}))

    with engine.connect() as c:
        recent = set(pd.read_sql(text("SELECT account_id FROM churn_interventions WHERE run_date >= :d"), c,
                                 params={"d": date.today() - timedelta(days=core.COOLDOWN_DAYS)}).account_id)
    cand = f[(f.risk_score >= core.RISK_THRESHOLD) & ~f.index.isin(recent)].copy()
    cand["priority"] = cand.risk_score * cand.arr
    cand = cand.sort_values("priority", ascending=False).head(limit or core.MAX_DAILY_FLAGS)

    run_id = f"R-{datetime.now():%Y%m%d%H%M%S}"
    rng = random.Random(run_id)
    print(f"{run_id}: {len(f)} active accounts scored, {len(cand)} flagged")

    for aid, r in cand.iterrows():
        base = dict(run_id=run_id, run_date=date.today(), account_id=aid, risk_score=round(float(r.risk_score), 4),
                    risk_tier=tier(r.risk_score), arr=float(r.arr))
        if rng.random() < core.HOLDOUT_RATE:  # control group for save-rate measurement
            _write(base | {"status": "holdout"})
            print(f"  {aid} holdout")
            continue
        ctx = RunContext(run_id=run_id, account_id=aid, contact_names=core.contact_names(aid))
        inputs = dict(account_id=aid, plan=r.plan, seats=int(r.seats), risk_score=base["risk_score"],
                      risk_tier=base["risk_tier"], model_facts=model_facts(r), schema_hint=core.AGENT_SCHEMA_HINT,
                      company_name=r.company, arr=f"{r.arr:,.0f}")
        try:
            profile, iv = run_account(ctx, inputs)
            profile = profile.model_copy(update={"risk_score": base["risk_score"], "risk_tier": base["risk_tier"]})
            issues = check_profile(profile, ctx) + check_intervention(iv, ctx)
            row = base | dict(profile_json=profile.model_dump_json(), intervention_json=iv.model_dump_json(),
                              guardrail_passed=not issues, guardrail_issues="; ".join(issues) or None,
                              status="guardrail_failed" if issues else "pending_review",
                              final_subject=iv.draft_email.subject, final_body=iv.draft_email.body)
        except Exception as e:
            row = base | {"status": "error", "error": f"{type(e).__name__}: {e}"[:2000]}
        _write(row)
        print(f"  {aid} {row['status']}")


# ── feedback loop ──────────────────────────────────────────────────────────────
def update_outcomes() -> None:
    today, n = pd.Timestamp(date.today()), 0
    with engine.begin() as c:
        rows = c.execute(text(
            "SELECT i.id, i.run_date, a.churn_date FROM churn_interventions i JOIN accounts a ON a.account_id = i.account_id "
            "WHERE i.outcome IS NULL AND i.status IN ('approved','edited','sent','holdout','rejected')")).mappings().all()
        for r in rows:
            rd = pd.Timestamp(r["run_date"])
            cd = pd.Timestamp(r["churn_date"]) if r["churn_date"] else None
            if cd is not None and rd <= cd <= rd + D(core.HORIZON_DAYS):
                outcome = "churned"
            elif today >= rd + D(core.HORIZON_DAYS):
                outcome = "retained"
            else:
                continue
            c.execute(update(interventions).where(interventions.c.id == r["id"]).values(outcome=outcome, outcome_at=date.today()))
            n += 1
    print(f"Labelled {n} outcomes.")


def learn() -> None:
    with engine.connect() as c:
        wins = pd.read_sql(text("SELECT id, account_id, profile_json, intervention_json, final_subject, final_body "
                                "FROM churn_interventions WHERE outcome = 'retained' AND status IN ('approved','edited','sent')"), c)
        have = set(pd.read_sql(text("SELECT entry_id FROM playbook"), c).entry_id)
        reasons = pd.read_sql(text("SELECT rejection_reason, COUNT(*) AS n FROM churn_interventions WHERE status = 'rejected' "
                                   "GROUP BY rejection_reason ORDER BY n DESC"), c)
    new = []
    for w in wins.itertuples():
        eid = f"PB-L{w.id:05d}"
        if eid in have:
            continue
        p, iv, names = json.loads(w.profile_json), json.loads(w.intervention_json), core.contact_names(w.account_id)
        new.append(dict(
            entry_id=eid, title=f"Past save: {iv['strategy']}", permits="", source="learned", created_at=datetime.now(),
            trigger=core.mask_pii(f"{p['likely_root_cause']}. Signals: " + "; ".join(s["detail"] for s in p["signals"]), names),
            actions=core.mask_pii(f"Strategy that retained the account: {iv['strategy']}. CSM-approved email:\n"
                                  f"Subject: {w.final_subject}\n{w.final_body}", names),
            constraints="Adapt to the current account. Same promise rules as the base playbook."))
    if new:
        with engine.begin() as c:
            c.execute(insert(playbook), new)
        core.index_playbook()
    print(f"Added {len(new)} learned playbook entries.")
    if not reasons.empty:
        print("Rejection reasons (turn frequent ones into guardrail rules or prompt fixes):")
        print(reasons.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seed", "init", "backtest", "train", "run", "outcomes", "learn"])
    ap.add_argument("--limit", type=int, help="max accounts for `run`")
    ap.add_argument("--accounts", type=int, default=200, help="synthetic accounts for `seed`")
    a = ap.parse_args()
    {"seed": lambda: core.seed_synthetic(a.accounts), "init": core.bootstrap, "backtest": backtest, "train": train,
     "run": lambda: run_daily(a.limit), "outcomes": update_outcomes, "learn": learn}[a.cmd]()
