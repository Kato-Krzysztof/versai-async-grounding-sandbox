# VersAI — Async Grounding Sandbox

Small `asyncio` prototype. A stream of shopping queries is processed by two agents:

- **Agent A (Personal Shopper)** drafts a recommendation + a structured claim.
- **Agent B (Grounding Judge)** checks that claim against a fixed inventory.

If a price / color / stock / product name doesn't match inventory, the verdict is
sent back to Agent A to fix. That loop is capped at 3 attempts; after that the task
is routed to a human-intervention queue.

No LangChain / LangGraph — just `asyncio` + `pydantic`.

Flow: `queue → Agent A → Agent B → grounded? finalize : retry (≤3) → human queue`

## Run

```
pip install -r requirements.txt
python -m versai.main
pytest
```

## Layout

```
versai/models.py        schemas + state enums
versai/inventory.py     source-of-truth product data
versai/agents.py        Agent A + Agent B
versai/orchestrator.py  queue + self-healing loop
versai/main.py          demo stream
tests/                  pytest-asyncio
```
