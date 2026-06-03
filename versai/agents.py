"""The two-agent swarm: Agent A (Personal Shopper) and Agent B (Grounding Judge)."""

from __future__ import annotations

import asyncio
import re
from typing import Callable, Optional

from .inventory import find_by_name, search
from .models import (
    AgentResponse,
    GroundingStatus,
    GroundingVerdict,
    GroundingViolation,
    Product,
    ProductClaim,
    UserQuery,
)

# A draft function maps (query, feedback, attempt) -> (message, claim). Injectable so
# tests and the demo can drive specific behaviours (clean, drifting, stubborn, ...).
DraftFn = Callable[[UserQuery, Optional[GroundingVerdict], int], "tuple[str, ProductClaim]"]


def _claim_of(p: Product) -> ProductClaim:
    return ProductClaim(name=p.name, price=p.price, color=p.color, in_stock=p.in_stock)


def _describe(p: Product) -> str:
    avail = "in stock" if p.in_stock else "currently out of stock"
    return f"I'd suggest the {p.name} by {p.brand} - {p.color}, ${p.price:,.2f}, {avail}."


def draft_recommendation(query, feedback, attempt):
    """Default mock generation: honest, and self-corrects from judge feedback."""
    if feedback and feedback.grounded_reference:        # asked to fix a previous answer
        p = feedback.grounded_reference
        return _describe(p), _claim_of(p)
    product = search(query.text)
    if product is None:
        return "I couldn't find a matching piece in our collection.", ProductClaim(name="<unknown>")
    return _describe(product), _claim_of(product)


class PersonalShopper:
    """Agent A. Stands in for a generative model; emits a checkable ProductClaim."""

    def __init__(self, generate: Optional[DraftFn] = None, latency: float = 0.0):
        self._generate = generate or draft_recommendation
        self._latency = latency

    async def respond(self, query: UserQuery, feedback=None, *, attempt: int = 1) -> AgentResponse:
        if self._latency:
            await asyncio.sleep(self._latency)        # simulate the LLM round-trip
        message, claim = self._generate(query, feedback, attempt)
        return AgentResponse(query_id=query.query_id, message=message, claim=claim, attempt=attempt)


_PRICE_RE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)")  # must start with a digit


def _money_eq(a: float, b: float) -> bool:
    return abs(a - b) < 0.01


class GroundingJudge:
    """Agent B. Deterministically validates a claim against inventory — no model calls."""

    def __init__(self, latency: float = 0.0):
        self._latency = latency

    async def evaluate(self, response: AgentResponse) -> GroundingVerdict:
        if self._latency:
            await asyncio.sleep(self._latency)
        return self._check(response)

    def _check(self, response: AgentResponse) -> GroundingVerdict:
        claim = response.claim
        product = find_by_name(claim.name)
        if product is None:
            v = GroundingViolation(field="name", claimed=claim.name,
                                   expected="a product in inventory",
                                   detail="No such product in inventory.")
            return self._regression(response, [v], grounded=None)

        violations: list[GroundingViolation] = []
        if claim.price is not None and not _money_eq(claim.price, product.price):
            violations.append(GroundingViolation(
                field="price", claimed=f"${claim.price:,.2f}", expected=f"${product.price:,.2f}",
                detail="Price does not match inventory."))
        if claim.color is not None and claim.color.strip().casefold() != product.color.casefold():
            violations.append(GroundingViolation(
                field="color", claimed=claim.color, expected=product.color,
                detail="Color does not match inventory."))
        if claim.in_stock is not None and claim.in_stock != product.in_stock:
            violations.append(GroundingViolation(
                field="in_stock", claimed=str(claim.in_stock), expected=str(product.in_stock),
                detail="Availability does not match inventory."))

        # Cross-check the prose: if it quotes prices, at least one must be the real
        # price (prose may legitimately mention sale/original/shipping figures too).
        quoted = []
        for raw in _PRICE_RE.findall(response.message):
            try:
                quoted.append(float(raw.replace(",", "")))
            except ValueError:
                continue
        if quoted and not any(_money_eq(q, product.price) for q in quoted):
            violations.append(GroundingViolation(
                field="message", claimed=", ".join(f"${q:,.2f}" for q in quoted),
                expected=f"${product.price:,.2f}",
                detail="No price quoted in the prose matches inventory."))

        if violations:
            return self._regression(response, violations, grounded=product)
        return GroundingVerdict(query_id=response.query_id, status=GroundingStatus.GROUNDED,
                                grounded_reference=product)

    def _regression(self, response, violations, grounded):
        return GroundingVerdict(
            query_id=response.query_id,
            status=GroundingStatus.HALLUCINATION_REGRESSION,
            violations=violations,
            correction_instructions=_correction_text(violations, grounded),
            grounded_reference=grounded,
        )


def _correction_text(violations, grounded: Optional[Product]) -> str:
    fields = ", ".join(v.field for v in violations)
    if grounded is None:
        return "Recommend only products that exist in the inventory."
    return (f"Your previous answer was wrong on: {fields}. "
            f"Re-issue the recommendation for '{grounded.name}' using exactly "
            f"price=${grounded.price:,.2f}, color={grounded.color}, in_stock={grounded.in_stock}.")
