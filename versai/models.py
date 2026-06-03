"""Pydantic models for state, agent handoffs, and grounding results."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskState(str, Enum):
    PENDING = "PENDING"
    SHOPPING = "SHOPPING"
    JUDGING = "JUDGING"
    CORRECTING = "CORRECTING"
    FINALIZED = "FINALIZED"
    HUMAN_INTERVENTION = "HUMAN_INTERVENTION"


class GroundingStatus(str, Enum):
    GROUNDED = "GROUNDED"
    HALLUCINATION_REGRESSION = "HALLUCINATION_REGRESSION"


class Product(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    brand: str
    price: float = Field(ge=0)
    color: str
    in_stock: bool


class UserQuery(BaseModel):
    query_id: str
    text: str


class ProductClaim(BaseModel):
    """What Agent A asserts about a product. Unset fields aren't checked."""

    name: str
    price: Optional[float] = None
    color: Optional[str] = None
    in_stock: Optional[bool] = None


class AgentResponse(BaseModel):
    query_id: str
    message: str
    claim: ProductClaim
    attempt: int = Field(default=1, ge=1)


class GroundingViolation(BaseModel):
    field: str
    claimed: str
    expected: str
    detail: str


class GroundingVerdict(BaseModel):
    query_id: str
    status: GroundingStatus
    violations: list[GroundingViolation] = Field(default_factory=list)
    correction_instructions: Optional[str] = None
    # On failure this carries the real record back to Agent A so it can self-correct.
    grounded_reference: Optional[Product] = None

    @property
    def is_grounded(self) -> bool:
        return self.status is GroundingStatus.GROUNDED


class TaskResult(BaseModel):
    query_id: str
    query_text: str
    final_state: TaskState
    attempts: int
    final_response: Optional[AgentResponse] = None
    final_verdict: Optional[GroundingVerdict] = None
    trajectory: list[str] = Field(default_factory=list)
