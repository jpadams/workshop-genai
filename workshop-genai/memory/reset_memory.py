"""
Remove all agent-memory data (and optionally schema) from the Neo4j database.

This deletes ONLY the agent-memory labels — your lesson knowledge graph
(Lesson, Chunk, Document, __KGBuilder__/__Entity__, Technology, ...) is left
untouched. The (:ToolCall)-[:RETRIEVED]->(lessonNode) edges are removed by the
DETACH DELETE of the ToolCall nodes; the lesson nodes themselves stay.

Usage:
    python workshop-genai/memory/reset_memory.py            # prompts to confirm
    python workshop-genai/memory/reset_memory.py --yes      # no prompt
    python workshop-genai/memory/reset_memory.py --keep-schema   # data only, keep indexes/constraints

By default it also drops the memory indexes/constraints (via the library's
name-scoped schema.drop_all(), which never touches lesson-graph schema). The
web/CLI agents recreate the memory schema automatically on next connect().
"""

import os
import sys
import asyncio

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from pydantic import SecretStr
from neo4j_agent_memory import MemoryClient, MemorySettings

# Every node label the agent-memory layer can create.
MEMORY_LABELS = [
    "Conversation", "Message", "Entity", "Preference", "Fact",
    "ReasoningTrace", "ReasoningStep", "ToolCall", "Tool",
    "User", "ConsolidationRun", "MemoryReadAudit",
]

_WHERE = " OR ".join(f"n:{label}" for label in MEMORY_LABELS)
COUNT_QUERY = f"MATCH (n) WHERE {_WHERE} RETURN count(n) AS c"
# Batched so it scales to large graphs (auto-commit / implicit transaction).
DELETE_QUERY = f"""
MATCH (n) WHERE {_WHERE}
CALL {{ WITH n DETACH DELETE n }} IN TRANSACTIONS OF 1000 ROWS
"""


def main():
    yes = "--yes" in sys.argv
    keep_schema = "--keep-schema" in sys.argv
    db = os.getenv("NEO4J_DATABASE") or "neo4j"

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    with driver.session(database=db) as session:
        before = session.run(COUNT_QUERY).single()["c"]
        print(f"Agent-memory nodes in '{db}': {before}")
        if before == 0 and keep_schema:
            print("Nothing to delete.")
            driver.close()
            return
        if not yes:
            resp = input(f"Delete these {before} memory nodes"
                         f"{'' if keep_schema else ' and drop the memory schema'}? [y/N] ")
            if resp.strip().lower() not in ("y", "yes"):
                print("Aborted.")
                driver.close()
                return
        # CALL ... IN TRANSACTIONS must run in auto-commit mode.
        session.run(DELETE_QUERY).consume()
        after = session.run(COUNT_QUERY).single()["c"]
        print(f"Deleted {before - after} nodes (remaining memory nodes: {after}).")

    driver.close()

    if not keep_schema:
        async def _drop():
            settings = MemorySettings(
                neo4j={
                    "uri": os.getenv("NEO4J_URI"),
                    "username": os.getenv("NEO4J_USERNAME"),
                    "password": SecretStr(os.getenv("NEO4J_PASSWORD") or ""),
                    "database": db,
                },
                embedding="openai/text-embedding-3-small",
                llm="openai/gpt-4o-mini",
            )
            async with MemoryClient(settings) as client:
                await client.schema.drop_all()
        asyncio.run(_drop())
        print("Dropped memory indexes/constraints (lesson-graph schema untouched).")

    print("Done.")


if __name__ == "__main__":
    main()
