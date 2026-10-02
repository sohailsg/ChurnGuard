# ChurnGuard

A human-in-the-loop, multi-agent system that watches product telemetry and support history, flags SaaS accounts likely to churn in the next 30 days, explains why, and drafts a personalized intervention grounded in your company's own retention playbook. Nothing is sent automatically — every recommendation lands in a review queue where a Customer Success Manager (CSM) approves, edits, or rejects it, and those decisions feed back into the system over time.

## The problem

Churn signals are scattered across usage logs, support tickets, and CSM intuition. By the time someone notices a drop in activity, the customer has often already decided to leave. Generic check-in emails don't help because they ignore the customer's actual friction.

What ChurnGuard delivers:

- **Earlier detection** — it scans every account daily instead of relying on a CSM to notice.
- **Explained risk** — each flag comes with the specific usage and ticket evidence behind it.
- **Specific outreach** — "your team stopped using Reports after the March export bug," not "just checking in."
- **A measurable loop** — a held-out control group and outcome tracking show whether the flags and drafts actually save accounts.

## How it works, end to end

```
Telemetry DB ──┐
Zendesk/DB    ─┼─► risk model (scikit-learn) ──► ranked, risk-scored accounts
CRM            ┘           │
                      top-N per day, minus a random holdout
                            │
                            ▼
              ┌── Data Analyst Agent ──┐    read-only SQL + ticket tools
              │   (CrewAI, Claude)     │    → Churn Risk Profile (JSON)
              └───────────┬────────────┘
                           ▼
              ┌── Customer Success Agent ──┐  retention-playbook RAG (ChromaDB)
              │   (CrewAI, Claude)         │  → strategy + draft email (JSON)
              └───────────┬────────────────┘
                           ▼
                 deterministic guardrails
          (citations, unapproved promises, PII/cross-customer leakage)
                           │
                    pass ──┴── fail → sent back to the agent to revise
                           │
                           ▼
                staging table (SQL) ──► Power BI dashboard ──► Flask review queue
                                                                 │
                                                CSM approves / edits / rejects
                                                                 │
                                                                 ▼
                                            outcome labels (30 days later) ──► feedback loop
                                          (successful saves → playbook, rejections → guardrail rules)
```

### 1. Risk scoring (not an agent)

A calibrated `HistGradientBoostingClassifier` scores every active account daily on engineered features — usage trend over 14-day windows, days since last activity, ticket volume and severity, CSAT, seat utilization, tenure, ARR. This is deliberately a plain ML model, not an LLM: it's cheap to run on the full account base, auditable, and backtestable against real churn history (`pipeline.py backtest`). The agents only get invoked for accounts the model already flagged — they explain and act on a risk score, they don't invent one.

The top accounts by `risk_score × ARR` go forward each day, capped at `MAX_DAILY_FLAGS`. A random slice (`HOLDOUT_RATE`, default 20%) is deliberately held out from intervention — this control group is what lets you measure whether outreach actually improves save rate, rather than just assuming it does.

### 2. The Data Analyst Agent — "who and why"

Given one account and the model's risk score, this agent investigates using two tools:
- **`read_only_sql`** — runs a single validated `SELECT` against telemetry. Queries are parsed with `sqlglot` and rejected if they touch any table outside `accounts`, `usage_daily`, `tickets`; select contact columns; reference another account; or aren't a plain `SELECT`. Every successful query gets a `query_ref` the agent must cite.
- **`recent_support_tickets`** — pulls the account's recent tickets (from the DB, or live from Zendesk if configured), with emails, phone numbers, and the contact's name masked before they ever reach the model.

It outputs a **Churn Risk Profile**: signals, each tied to specific `query_ref`/ticket-ID evidence, plus a likely root cause, sentiment, and confidence. A guardrail rejects the output and sends it back to the agent if any cited evidence wasn't actually returned by a tool — so hallucinated reasons can't make it through.

### 3. The Customer Success Agent — "how to fix it"

Given the profile, this agent searches a **retention playbook** (seeded with ten real playbook entries, indexed in ChromaDB) for the best-fitting strategy, then drafts a short email. It outputs an **Intervention Package**: strategy, playbook citations, and a draft addressed with placeholders (`[CONTACT_FIRST_NAME]`, `[CSM_NAME]`) instead of real names.

### 4. Guardrails (deterministic, not another LLM call)

Before anything reaches a human, Python code checks:
- every cited playbook entry was actually returned by the search tool
- no discount, roadmap, or guarantee language unless the cited playbook entry's `permits` field allows it
- no email addresses, phone numbers, real contact names, other account IDs, or other customers' names
- no leaked internal IDs (query refs, ticket IDs, playbook IDs) in the customer-facing text
- draft length and placeholder sanity
- any custom rules a CSM has added from the review UI after seeing a bad draft

A failing check sends the specific list of problems back to the agent as a CrewAI task guardrail, so it gets one or more chances to self-correct before the row is marked `guardrail_failed` for human review.

### 5. Review queue

A Flask app shows each flagged account's evidence, strategy, and editable draft side by side. A CSM approves, edits (edits are themselves re-checked against the guardrails), or rejects with a reason. Approved drafts never auto-send — the app hands off to the CSM's own mail client. Power BI reads three SQL views directly from the staging table: the live queue, daily KPIs (approval rate, edit distance, volume), and outcomes by arm (intervened vs. holdout vs. not actioned).

### 6. Feedback loop

Thirty days after each flag, `pipeline.py outcomes` labels whether the account churned or was retained, using the account's actual churn date. `pipeline.py learn` then:
- turns retained accounts that got an approved/edited intervention into new playbook entries (PII-masked), so future drafts can draw on what actually worked
- surfaces the most common rejection reasons, so recurring CSM objections become new guardrail rules or prompt fixes instead of staying tribal knowledge

## Success metrics

- **Precision** of flagged accounts (flagged → actually churned or downgraded without intervention)
- **Save rate**: intervened accounts vs. the holdout control group
- **CSM approval rate** and **edit distance** on drafts
- **Time from risk onset to first outreach**

All four are computed directly from the `v_kpi_daily` and `v_outcomes` SQL views.

## Project layout

| File | Purpose |
|---|---|
| `core.py` | Settings, SQL schema, Power BI views, PII masking, retention playbook (seed data + ChromaDB indexing), synthetic data generator |
| `schemas.py` | Pydantic data contracts (`RiskProfile`, `Intervention`) shared between the two agents |
| `tools.py` | Agent tools: validated read-only SQL, ticket fetch (DB or Zendesk), playbook RAG search |
| `guardrails.py` | Deterministic checks: grounding, unapproved promises, PII/cross-customer leakage |
| `crew.py` | CrewAI agent and task definitions, wired to the schemas and guardrails |
| `pipeline.py` | CLI: feature engineering, model training, backtest, daily run, outcome labeling, learning loop |
| `review_app.py` | Flask CSM review queue |
| `requirements.txt` | Dependencies |

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python pipeline.py seed        # synthetic demo data + playbook (use `init` on real data)
python pipeline.py backtest    # validate the risk model against real churn history first
python pipeline.py train
python pipeline.py run --limit 3   # score accounts, run the agents, write to the staging table
python review_app.py              # CSM review queue (Flask, http://127.0.0.1:5000)

# run on a schedule (e.g. daily cron) and weekly:
python pipeline.py outcomes    # label 30-day outcomes
python pipeline.py learn       # saves -> playbook, rejections -> report
```

Point Power BI at the `v_review_queue`, `v_kpi_daily`, and `v_outcomes` views in the same database.

## Design choices worth calling out

- **No send capability in v1.** The agents can only write to a staging table; sending is a manual action by the CSM in their own mail client. This is a deliberate scope limit, not a missing feature.
- **A plain ML model does the scoring, not an LLM.** Agents are reserved for the parts that genuinely need reasoning over unstructured evidence (why is this account at risk, what should we say) — not for a task a calibrated classifier already does better and cheaper.
- **Guardrails are code, not a third agent.** Promise-checking, citation-checking, and leakage-checking are deterministic and fast, which also makes them auditable and easy for a CSM to extend with their own regex rules from the UI.
- **The holdout group is the backbone of the evaluation.** Without it, "accounts we intervened on were retained" can't be distinguished from "accounts we intervened on would have stayed anyway."
