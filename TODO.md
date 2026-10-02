# ChurnGuard: remaining work

## Done so far
- Python 3.11 venv at `C:\Users\cd035\churnguard-venv` (outside the project, because venv refuses paths containing `;`). Dependencies installed.
- Offline pipeline verified: `seed` → `backtest` (ROC AUC 0.921 vs 0.904 for the simple rule) → `train`.
- `review_app.py` rewritten from Streamlit to Flask (`python review_app.py` → http://127.0.0.1:5000). README and `requirements.txt` updated.
- `core.py` / `crew.py` accept `LLM_BASE_URL`, so any provider can be used (Ollama, etc.) through env vars.
- 51 offline tests in `tests/` pass (SQL validation, guardrails, features and labels, pipeline with a stubbed crew, outcomes and learn, tools, Flask UI).

## 1. Get a working LLM (blocked)
No Anthropic key. Chosen path: local Ollama with `qwen2.5:7b` (~4.7 GB). The pull was interrupted twice and the model is **not installed**.
- [ ] `ollama pull qwen2.5:7b` (run it yourself with `! ollama pull qwen2.5:7b` if you want to watch progress).
- [ ] Note: the existing `gemma4:e4b` is broken (its model file is missing on disk). Either delete it with `ollama rm gemma4:e4b` or re-pull it.
- [ ] Alternatives if qwen is too weak or slow: a `:cloud` model (needs `ollama signin`; data goes to ollama.com), or another provider's API key.

## 2. Live agent run
Set these in your shell, then run:
```
set ANALYST_MODEL=ollama/qwen2.5:7b
set CSM_MODEL=ollama/qwen2.5:7b
set LLM_BASE_URL=http://localhost:11434
set CREW_VERBOSE=1
python pipeline.py run --limit 3
```
- [ ] Check staged rows: statuses are `pending_review`, `guardrail_failed` or `holdout`, not `error`.
- [ ] Cited evidence IDs resolve in `agent_query_log`; drafts have no PII and use the placeholders.
- [ ] If `error` rows appear, read the `error` column. Likely fixes are in `crew.py` (`_as`, structured output parsing) or the prompts. Small models often fail the strict JSON and guardrail loop, so you may need more guardrail retries or simpler prompts.

## 3. Review UI check
- [ ] `python review_app.py`, open the page in a browser, and approve, edit, reject and mark one row as sent.
- [ ] Add a guardrail rule and confirm a draft violating it is blocked.
- [ ] The UI is unit-tested with Flask's test client but has not been viewed in a browser, so check layout and dark mode.

## 4. Feedback loop
- [ ] After approving rows, run `python pipeline.py outcomes` and `python pipeline.py learn`. Outcomes only label rows 30 days old, so use the test in `tests/test_pipeline_flow.py` as the reference, or back-date `run_date` by hand.

## 5. Cleanup and hardening
- [ ] `guardrails.draft_issues` accepts 50–260 words but the prompt and error text say 80–200. Align them.
- [ ] `guardrails._account_names` is `lru_cache`d and goes stale after re-seeding in a long-lived process.
- [ ] `review_app.py` uses a hard-coded Flask `secret_key`. Fine for local use; change it before any shared deployment, and add auth.
- [ ] The read-only SQLite URI in `core.py` is relative to the working directory. Run commands from the project folder or set `DATABASE_URL` to an absolute path.
- [ ] Add `.env.example` listing `DATABASE_URL`, `ANALYST_MODEL`, `CSM_MODEL`, `LLM_BASE_URL`, `RISK_THRESHOLD`, `MAX_DAILY_FLAGS`, `HOLDOUT_RATE`, and the optional Zendesk variables.
- [ ] Add a README "Testing" section: `python -m pytest -q` from the project folder with the venv Python.
- [ ] Add `tests/__init__.py` cleanup or switch to `rootdir` imports, since the UI test imports `tests.test_pipeline_flow`.
- [ ] Initialise git and add a `.gitignore` for `churnguard.db`, `chroma/`, `churn_model.joblib`, `__pycache__/`.

## 6. Before using real data
- [ ] Use `python pipeline.py init` (not `seed`) and load real accounts, usage and tickets.
- [ ] Run `backtest` on real churn history and review precision at the daily cap before trusting scores.
- [ ] On Postgres, set `AGENT_DATABASE_URL` to a role with SELECT-only grants on `accounts`, `usage_daily` and `tickets`.
- [ ] Point Power BI at `v_review_queue`, `v_kpi_daily` and `v_outcomes`.
- [ ] Schedule `run` daily and `outcomes` / `learn` weekly.
