"""Demo: push a concurrent stream of queries through the swarm. Run: python -m versai.main"""

from __future__ import annotations

import asyncio
import time

from .agents import GroundingJudge, PersonalShopper, draft_recommendation
from .models import ProductClaim, UserQuery
from .orchestrator import Orchestrator

# A burst of queries arriving "at once". Each id maps to a scripted behaviour below so a
# single shopper can act out every path: clean / out-of-stock / self-heal / regressions.
STREAM = [
    UserQuery(query_id="q-clean",    text="show me the monogram leather tote"),
    UserQuery(query_id="q-oos",      text="do you have the cashmere knit sweater?"),
    UserQuery(query_id="q-heal",     text="is the leather tote on sale?"),
    UserQuery(query_id="q-stubborn", text="how much is the silk trench coat?"),
    UserQuery(query_id="q-unknown",  text="something dramatic for the opera"),
]


def scripted_shopper(query, feedback, attempt):
    """Mock LLM, scripted per query id to exercise each grounding path."""
    # When the judge hands back the truth, fix it - except the deliberately stubborn one.
    if feedback and feedback.grounded_reference and query.query_id != "q-stubborn":
        p = feedback.grounded_reference
        return (f"Apologies - corrected: the {p.name} is {p.color}, ${p.price:,.0f}.",
                ProductClaim(name=p.name, price=p.price, color=p.color, in_stock=p.in_stock))

    if query.query_id == "q-heal":        # hallucinate a sale price on the first pass
        return ("Great news - it's on sale for $1,500!",
                ProductClaim(name="Monogram Leather Tote", price=1500.0, color="Noir", in_stock=True))
    if query.query_id == "q-stubborn":    # insists on a fantasy price, never corrects
        return ("Trust me, the trench coat is just $300.",
                ProductClaim(name="Classic Silk Trench Coat", price=300.0, color="Beige", in_stock=True))
    if query.query_id == "q-unknown":     # recommends a product that doesn't exist
        return ("You'll adore the Velvet Opera Cape.",
                ProductClaim(name="Velvet Opera Cape", price=3200.0))

    return draft_recommendation(query, feedback, attempt)   # honest, grounded path


def _summary(verdict) -> str:
    if verdict is None or verdict.is_grounded:
        return "grounded"
    return "; ".join(f"{v.field} ({v.claimed} vs {v.expected})" for v in verdict.violations)


def _drain(queue) -> list:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


async def main() -> None:
    orchestrator = Orchestrator(
        shopper=PersonalShopper(generate=scripted_shopper, latency=0.02),
        judge=GroundingJudge(latency=0.02),
    )

    started = time.perf_counter()
    results = await orchestrator.run(STREAM, workers=4, stream_delay=0.02)
    elapsed_ms = (time.perf_counter() - started) * 1000
    results.sort(key=lambda r: r.query_id)

    print(f"\n=== VersAI grounding sandbox - {len(results)} queries / 4 workers "
          f"in {elapsed_ms:.0f} ms ===\n")
    for r in results:
        name = r.final_response.claim.name if r.final_response else "-"
        print(f"[{r.query_id:<10}] {r.final_state.value:<18} attempts={r.attempts}  "
              f"{name!r:<28} -> {_summary(r.final_verdict)}")

    escalations = _drain(orchestrator.human_review)
    print(f"\nHuman-intervention queue ({len(escalations)}):")
    for r in escalations:
        print(f"  - {r.query_id}: {_summary(r.final_verdict)}")

    healed = next((r for r in results if r.query_id == "q-heal"), None)
    if healed:
        print(f"\nTrajectory for {healed.query_id} (self-heal):")
        for step in healed.trajectory:
            print(f"  {step}")


if __name__ == "__main__":
    asyncio.run(main())
