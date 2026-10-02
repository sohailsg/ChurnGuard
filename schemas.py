"""Data contracts between agents (strict, machine-checkable)."""
from typing import Literal

from pydantic import BaseModel, Field


class Signal(BaseModel):
    type: Literal["usage", "support", "engagement", "commercial"]
    detail: str = Field(description="Specific, quantified observation, e.g. 'Reports sessions down 46% (last 14d vs prior 14d)'")
    evidence: list[str] = Field(min_length=1, description="query_ref values (Q-...) and/or ticket IDs (T-...) returned by your tools")


class RiskProfile(BaseModel):
    account_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_tier: Literal["high", "medium", "low"]
    signals: list[Signal] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "neutral", "frustrated", "angry", "unknown"]
    likely_root_cause: str
    confidence: Literal["low", "medium", "high"]


class DraftEmail(BaseModel):
    subject: str
    body: str = Field(description="80-200 words. Greet with [CONTACT_FIRST_NAME], sign off as [CSM_NAME].")


class Intervention(BaseModel):
    account_id: str
    strategy: str = Field(description="Short name of the save strategy")
    rationale: str = Field(description="One or two sentences linking the strategy to the profile evidence")
    playbook_refs: list[str] = Field(min_length=1, description="Entry IDs returned by retention_playbook_search")
    draft_email: DraftEmail
    requires_approval: Literal[True] = True
