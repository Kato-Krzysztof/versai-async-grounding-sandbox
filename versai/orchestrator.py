"""Async orchestrator: ingestion queue, worker pool, and the bounded self-healing loop."""

from __future__ import annotations

import asyncio
from typing import Optional

from .agents import GroundingJudge, PersonalShopper
from .models import GroundingVerdict, TaskResult, TaskState, UserQuery

MAX_ATTEMPTS = 3


class Orchestrator:
    """Drives queries through Agent A -> Agent B, self-healing up to MAX_ATTEMPTS."""

    def __init__(self, shopper: Optional[PersonalShopper] = None,
                 judge: Optional[GroundingJudge] = None,
                 max_attempts: int = MAX_ATTEMPTS):
        self.shopper = shopper or PersonalShopper()
        self.judge = judge or GroundingJudge()
        self.max_attempts = max_attempts
        self.ingestion: asyncio.Queue[UserQuery] = asyncio.Queue()
        self.results: asyncio.Queue[TaskResult] = asyncio.Queue()
        self.human_review: asyncio.Queue[TaskResult] = asyncio.Queue()

    async def submit(self, query: UserQuery) -> None:
        await self.ingestion.put(query)

    async def _process(self, query: UserQuery) -> TaskResult:
        """One query's full trajectory: shop -> judge -> (finalize | correct), bounded."""
        trajectory = [f"{TaskState.PENDING.value} q={query.query_id}"]
        feedback: Optional[GroundingVerdict] = None
        response = verdict = None

        for attempt in range(1, self.max_attempts + 1):
            trajectory.append(f"{TaskState.SHOPPING.value} attempt={attempt}")
            response = await self.shopper.respond(query, feedback=feedback, attempt=attempt)

            trajectory.append(f"{TaskState.JUDGING.value} claim={response.claim.name!r}")
            verdict = await self.judge.evaluate(response)

            if verdict.is_grounded:
                trajectory.append(f"{TaskState.FINALIZED.value} attempt={attempt}")
                return self._result(query, TaskState.FINALIZED, attempt, response, verdict, trajectory)

            trajectory.append(f"{TaskState.CORRECTING.value} bad={[v.field for v in verdict.violations]}")
            feedback = verdict  # hand the error payload back to Agent A for the next attempt

        # Loop guard tripped: never finalized within the attempt budget.
        trajectory.append(f"{TaskState.HUMAN_INTERVENTION.value} after={self.max_attempts}")
        result = self._result(query, TaskState.HUMAN_INTERVENTION, self.max_attempts,
                              response, verdict, trajectory)
        await self.human_review.put(result)
        return result

    @staticmethod
    def _result(query, state, attempts, response, verdict, trajectory) -> TaskResult:
        return TaskResult(query_id=query.query_id, query_text=query.text, final_state=state,
                          attempts=attempts, final_response=response, final_verdict=verdict,
                          trajectory=trajectory)

    async def _worker(self) -> None:
        while True:
            query = await self.ingestion.get()
            try:
                await self.results.put(await self._process(query))
            except Exception as exc:  # one bad task must not take down the worker
                failed = TaskResult(
                    query_id=query.query_id, query_text=query.text,
                    final_state=TaskState.HUMAN_INTERVENTION, attempts=0,
                    trajectory=[f"{TaskState.HUMAN_INTERVENTION.value} error={type(exc).__name__}: {exc}"],
                )
                await self.human_review.put(failed)
                await self.results.put(failed)
            finally:
                self.ingestion.task_done()

    async def run(self, queries, workers: int = 4) -> list[TaskResult]:
        """Submit queries, process them concurrently, and return every result."""
        for query in queries:
            await self.submit(query)

        pool = [asyncio.create_task(self._worker()) for _ in range(workers)]
        await self.ingestion.join()          # block until every query is done
        for task in pool:
            task.cancel()
        await asyncio.gather(*pool, return_exceptions=True)
        return _drain(self.results)


def _drain(queue: asyncio.Queue) -> list:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items
