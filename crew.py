"""CrewAI agents + tasks. One crew run per account."""
from crewai import LLM, Agent, Crew, Process, Task

from core import ANALYST_MODEL, CREW_VERBOSE, CSM_MODEL, LLM_BASE_URL
from guardrails import _strip_fences, as_task_guardrail, check_intervention, check_profile
from schemas import Intervention, RiskProfile
from tools import PlaybookSearchTool, ReadOnlySQLTool, RecentTicketsTool, RunContext

PROFILE_TASK = """Investigate account {account_id} ({plan} plan, {seats} seats).
A calibrated churn model scored it {risk_score} ({risk_tier} risk) for churn within 30 days. Model inputs (context only, not citable):
{model_facts}

Steps:
1. Use read_only_sql to confirm the usage trend: compare the last 14 days with the previous 14 days per feature, and check daily active users. Every query must filter account_id = '{account_id}'.
2. Use recent_support_tickets to read the last 60 days of tickets.
3. Write 2-5 signals. Each signal's evidence list must contain the query_ref values (Q-...) and/or ticket IDs (T-...) that support it. Never cite anything a tool did not return.
4. Infer the most likely root cause and the customer's sentiment from the evidence only. If evidence is thin, say so and set confidence to low.

Tables you can query:
{schema_hint}

Copy risk_score and risk_tier exactly as given."""

INTERVENE_TASK = """Using the Churn Risk Profile for account {account_id} ({company_name}, {plan} plan, ARR {arr}), design a save strategy and draft one outreach email from the CSM.

1. Call retention_playbook_search at least twice: once for the root cause and once for the strongest other signal.
2. Pick the best-fitting strategy. playbook_refs must contain only entry IDs returned by the search.
3. Draft a 50-260 word email to the customer's main contact. Greet with [CONTACT_FIRST_NAME] and sign off as [CSM_NAME]. Name the specific friction from the profile in plain customer language (no internal metrics, scores or IDs).
4. Do not mention discounts, refunds, credits, roadmap or release dates, or guarantees unless a cited entry's permits allow it, and then only within its constraints.
5. Never include email addresses, phone numbers, real names or other customers."""


def _as(task_output, model):
    return task_output.pydantic or model.model_validate_json(_strip_fences(task_output.raw))


def run_account(ctx: RunContext, inputs: dict) -> tuple[RiskProfile, Intervention]:
    analyst = Agent(
        role="Senior SaaS Data Analyst",
        goal="Explain, with cited evidence, why an account is at risk of churning within 30 days",
        backstory="You trust only query results and ticket data. You never guess; every claim points to a query_ref or ticket ID.",
        tools=[ReadOnlySQLTool(ctx), RecentTicketsTool(ctx)],
        llm=LLM(model=ANALYST_MODEL, temperature=0, base_url=LLM_BASE_URL), allow_delegation=False, max_iter=15, verbose=CREW_VERBOSE)

    csm = Agent(
        role="Enterprise Customer Success Manager",
        goal="Turn a churn risk profile into a specific, empathetic save plan grounded in the retention playbook",
        backstory="You write like a senior CSM: concrete, brief, warm. You never promise anything the playbook does not permit.",
        tools=[PlaybookSearchTool(ctx)],
        llm=LLM(model=CSM_MODEL, temperature=0.4, base_url=LLM_BASE_URL), allow_delegation=False, max_iter=10, verbose=CREW_VERBOSE)

    profile_task = Task(
        description=PROFILE_TASK, expected_output="A Churn Risk Profile JSON object", agent=analyst,
        output_pydantic=RiskProfile, guardrail=as_task_guardrail(check_profile, ctx, RiskProfile))

    intervene_task = Task(
        description=INTERVENE_TASK, expected_output="An Intervention Package JSON object", agent=csm,
        context=[profile_task], output_pydantic=Intervention,
        guardrail=as_task_guardrail(check_intervention, ctx, Intervention))

    result = Crew(agents=[analyst, csm], tasks=[profile_task, intervene_task],
                  process=Process.sequential, verbose=CREW_VERBOSE).kickoff(inputs=inputs)
    return _as(result.tasks_output[0], RiskProfile), _as(result.tasks_output[1], Intervention)
