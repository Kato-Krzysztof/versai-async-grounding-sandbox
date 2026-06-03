"""End-to-end tests for the async grounding swarm.

Uses the real GroundingJudge and Orchestrator; only Agent A's generation is injected
(the production seam, PersonalShopper(generate=fn)). No sleeps, so the suite is
deterministic. asyncio_mode="auto" is set in pyproject, so no per-test marker.
"""

from __future__ import annotations

from versai.agents import GroundingJudge, PersonalShopper, draft_recommendation
from versai.inventory import find_by_name
from versai.models import (
    AgentResponse,
    GroundingStatus,
    ProductClaim,
    TaskState,
    UserQuery,
)
from versai.orchestrator import Orchestrator

TOTE = "Monogram Leather Tote"     # $1850, Noir, in stock
SWEATER = "Cashmere Knit Sweater"  # $950, Camel, out of stock


def _claim(product) -> ProductClaim:
    return ProductClaim(name=product.name, price=product.price,
                        color=product.color, in_stock=product.in_stock)


def clean_gen(name):
    """Agent A that grounds correctly on the first try."""
    def gen(query, feedback, attempt):
        p = find_by_name(name)
        return f"The {p.name} by {p.brand} -- {p.color}, ${p.price:,.0f}.", _claim(p)
    return gen


def heal_gen(name):
    """Wrong price on attempt 1, then heals from the judge's feedback."""
    def gen(query, feedback, attempt):
        if feedback and feedback.grounded_reference:
            return draft_recommendation(query, feedback, attempt)
        real = find_by_name(name)
        bad = ProductClaim(name=name, price=0.01, color=real.color, in_stock=real.in_stock)
        return f"The {name}, a steal at $0.01.", bad
    return gen


def stubborn_gen(name):
    """Always hallucinates, ignoring feedback -> never grounds."""
    def gen(query, feedback, attempt):
        real = find_by_name(name)
        bad = ProductClaim(name=name, price=1.0, color="Neon", in_stock=not real.in_stock)
        return f"Buy the {name}, just $1.00 and in stock!", bad
    return gen


async def test_clean_query_finalizes_in_one_attempt():
    orch = Orchestrator(shopper=PersonalShopper(generate=clean_gen(TOTE)))
    [result] = await orch.run([UserQuery(query_id="q-clean", text="the tote")], workers=4)

    assert result.final_state is TaskState.FINALIZED
    assert result.attempts == 1
    assert result.final_verdict.is_grounded
    assert result.final_verdict.violations == []
    assert result.trajectory[0].startswith(TaskState.PENDING.value)
    assert result.trajectory[-1].startswith(TaskState.FINALIZED.value)
    assert not any(s.startswith(TaskState.CORRECTING.value) for s in result.trajectory)
    assert orch.human_review.empty()


async def test_judge_intercepts_price_and_out_of_stock_hallucination():
    real = find_by_name(SWEATER)
    assert real.price == 950.00 and real.in_stock is False  # premise

    claim = ProductClaim(name=SWEATER, price=499.00, in_stock=True)
    response = AgentResponse(query_id="q", message="Grab the sweater!", claim=claim, attempt=1)
    verdict = await GroundingJudge().evaluate(response)

    assert verdict.status is GroundingStatus.HALLUCINATION_REGRESSION
    fields = {v.field for v in verdict.violations}
    assert "price" in fields and "in_stock" in fields
    price_v = next(v for v in verdict.violations if v.field == "price")
    assert price_v.claimed == "$499.00" and price_v.expected == "$950.00"
    # Judge hands back the truth so Agent A can self-heal.
    assert verdict.grounded_reference.name == SWEATER
    assert verdict.correction_instructions is not None

    # End to end: drift once, then heal -> FINALIZED in 2 attempts.
    orch = Orchestrator(shopper=PersonalShopper(generate=heal_gen(SWEATER)))
    [healed] = await orch.run([UserQuery(query_id="q-heal", text="the sweater")], workers=2)
    assert healed.final_state is TaskState.FINALIZED
    assert healed.attempts == 2
    assert any(s.startswith(TaskState.CORRECTING.value) for s in healed.trajectory)


async def test_stubborn_shopper_escalates_to_human_after_max_attempts():
    orch = Orchestrator(shopper=PersonalShopper(generate=stubborn_gen(TOTE)), max_attempts=3)
    [result] = await orch.run([UserQuery(query_id="q-stub", text="the tote")], workers=4)

    assert result.final_state is TaskState.HUMAN_INTERVENTION
    assert result.attempts == 3
    assert not result.final_verdict.is_grounded

    assert orch.human_review.qsize() == 1
    assert orch.human_review.get_nowait() is result  # same object on both queues

    assert sum(s.startswith(TaskState.SHOPPING.value) for s in result.trajectory) == 3
    assert sum(s.startswith(TaskState.CORRECTING.value) for s in result.trajectory) == 3
    assert result.trajectory[-1].startswith(TaskState.HUMAN_INTERVENTION.value)
    assert not any(s.startswith(TaskState.FINALIZED.value) for s in result.trajectory)


async def test_attempt_counter_is_exact():
    cases = [
        ("q1", clean_gen(TOTE), TaskState.FINALIZED, 1),
        ("q2", heal_gen(TOTE), TaskState.FINALIZED, 2),
        ("q3", stubborn_gen(TOTE), TaskState.HUMAN_INTERVENTION, 3),
    ]
    for qid, gen, state, attempts in cases:
        orch = Orchestrator(shopper=PersonalShopper(generate=gen))
        [r] = await orch.run([UserQuery(query_id=qid, text="anything")])
        assert (r.final_state, r.attempts) == (state, attempts), qid


async def test_worker_pool_processes_mixed_batch():
    clean_ids = {f"clean-{i}" for i in range(6)}
    heal_ids = {f"heal-{i}" for i in range(5)}
    stub_ids = {f"stub-{i}" for i in range(4)}
    real = find_by_name(TOTE)

    def gen(query, feedback, attempt):
        qid = query.query_id
        if qid in clean_ids:
            return draft_recommendation(query, feedback, attempt)
        if qid in heal_ids:
            if feedback and feedback.grounded_reference:
                return draft_recommendation(query, feedback, attempt)
            return f"The {real.name} for $7.77.", ProductClaim(
                name=real.name, price=7.77, color=real.color, in_stock=real.in_stock)
        return f"The {real.name} for $2, in stock!", ProductClaim(
            name=real.name, price=2.0, color="Neon", in_stock=not real.in_stock)

    orch = Orchestrator(shopper=PersonalShopper(generate=gen), max_attempts=3)
    all_ids = clean_ids | heal_ids | stub_ids
    results = await orch.run([UserQuery(query_id=q, text="tote") for q in all_ids], workers=4)

    assert {r.query_id for r in results} == all_ids  # nothing dropped or duplicated
    by_id = {r.query_id: r for r in results}
    assert all(by_id[q].final_state is TaskState.FINALIZED and by_id[q].attempts == 1 for q in clean_ids)
    assert all(by_id[q].final_state is TaskState.FINALIZED and by_id[q].attempts == 2 for q in heal_ids)
    assert all(by_id[q].final_state is TaskState.HUMAN_INTERVENTION and by_id[q].attempts == 3 for q in stub_ids)

    escalated = set()
    while not orch.human_review.empty():
        escalated.add(orch.human_review.get_nowait().query_id)
    assert escalated == stub_ids
    assert orch.results.empty()


async def test_outcome_independent_of_worker_count():
    queries = [UserQuery(query_id=f"q-{i}", text="tote") for i in range(12)]
    outcomes = {}
    for n in (1, 8):
        orch = Orchestrator(shopper=PersonalShopper(generate=clean_gen(TOTE)))
        results = await orch.run(list(queries), workers=n)
        outcomes[n] = {r.query_id: (r.final_state, r.attempts) for r in results}
    assert outcomes[1] == outcomes[8]
    assert all(v == (TaskState.FINALIZED, 1) for v in outcomes[1].values())


# Grounding judge: each violation in isolation, plus what it must not flag.

async def _verdict(claim, message="A fine choice."):
    return await GroundingJudge().evaluate(
        AgentResponse(query_id="q", message=message, claim=claim, attempt=1))


async def test_judge_flags_color_mismatch():
    verdict = await _verdict(ProductClaim(name=TOTE, color="Beige"))
    assert verdict.status is GroundingStatus.HALLUCINATION_REGRESSION
    color_v = next(v for v in verdict.violations if v.field == "color")
    assert color_v.claimed == "Beige" and color_v.expected == "Noir"


async def test_judge_flags_unknown_product_name():
    verdict = await _verdict(ProductClaim(name="Velvet Opera Cape", price=3200.0))
    assert verdict.status is GroundingStatus.HALLUCINATION_REGRESSION
    assert {v.field for v in verdict.violations} == {"name"}
    assert verdict.grounded_reference is None


async def test_judge_detects_price_drift_in_prose():
    claim = _claim(find_by_name(TOTE))  # claim is correct; only the prose drifts
    verdict = await _verdict(claim, message="Limited time: just $99!")
    assert verdict.status is GroundingStatus.HALLUCINATION_REGRESSION
    assert {v.field for v in verdict.violations} == {"message"}


async def test_honest_out_of_stock_is_grounded():
    verdict = await _verdict(_claim(find_by_name(SWEATER)), message="The sweater is lovely.")
    assert verdict.is_grounded
    assert verdict.violations == []


async def test_money_comparison_tolerates_float_noise():
    real = find_by_name(TOTE)
    assert (await _verdict(ProductClaim(name=TOTE, price=real.price + 0.004))).is_grounded
    off = await _verdict(ProductClaim(name=TOTE, price=real.price + 0.5))
    assert any(v.field == "price" for v in off.violations)
