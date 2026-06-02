"""
Text2Cypher agent with Neo4j Agent Memory (context-graph style).

This extends ``agent_text2cypher.py`` by wiring in the ``neo4j-agent-memory``
package so that, in the *same* Neo4j database as the lesson knowledge graph, we
also build a **context graph** of the human <-> agent interaction:

- Short-term memory  -> (:Conversation)-[:HAS_MESSAGE]->(:Message)  (the chat history)
- Long-term memory   -> (:Entity) / (:Preference) / (:Fact)         (auto-extracted by an LLM)
- Reasoning memory   -> (:ReasoningTrace)->(:ReasoningStep)->(:ToolCall)
                        i.e. *how the graph was searched over time*: which tool ran,
                        with what query, what it returned, how long it took, success/failure.

The memory uses its own labels so it coexists with the lesson graph, and because
it lives in the same database you can Cypher-join conversation-derived entities to
lesson content.

Run:
    python workshop-genai/memory/agent_memory.py

Inspect the resulting context graph (Neo4j Browser, or the neo4j-cli `:schema`):
    // See all memory labels/relationships next to the lesson graph
    CALL db.schema.visualization()

    // How the graph has been searched over time (one row per tool call)
    MATCH (t:ReasoningTrace)-[:HAS_STEP]->(:ReasoningStep)-[:USES_TOOL]->(c:ToolCall)
    RETURN t.started_at AS at, t.task AS task,
           c.tool_name AS tool, c.arguments AS args,
           c.status AS status, c.duration_ms AS ms
    ORDER BY at

    // The conversation as a context graph: messages and the entities they mention
    MATCH (:Conversation {session_id: 'workshop-session-001'})-[:HAS_MESSAGE]->(m:Message)
    OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
    RETURN m.role AS role, m.content AS content, collect(e.name) AS entities

Relationship shapes created by the memory layer (verified):
    (:Conversation)-[:HAS_MESSAGE|FIRST_MESSAGE]->(:Message)-[:NEXT_MESSAGE]->(:Message)
    (:Message)-[:MENTIONS]->(:Entity)            // long-term context graph
    (:ReasoningTrace)-[:HAS_STEP]->(:ReasoningStep)-[:USES_TOOL]->(:ToolCall)-[:INSTANCE_OF]->(:Tool)

Note: LLM-extracted :Entity nodes are stored without embeddings, so they are
reachable by Cypher / MENTIONS traversal but not by vector search_entities().
"""

import os
import asyncio
import contextvars

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

# tag::import_memory[]
from pydantic import SecretStr
from neo4j_agent_memory import MemoryClient, MemorySettings
from neo4j_agent_memory.memory.reasoning import StreamingTraceRecorder, ToolCallStatus
# end::import_memory[]

# Initialize the chat model (the agent's "reasoning" LLM)
model = init_chat_model("gpt-5.2", model_provider="openai")

# Connect to Neo4j database (raw driver used by the neo4j-graphrag retrievers)
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Define retrieval query
retrieval_query = """
MATCH (node)-[:FROM_DOCUMENT]->(d)-[:PDF_OF]->(lesson)
RETURN
    node.text as text, score,
    lesson.url as lesson_url,
    collect {
        MATCH (node)<-[:FROM_CHUNK]-(entity)-[r]->(other)-[:FROM_CHUNK]->()
        WITH toStringList([
            [l IN labels(entity)
                WHERE NOT l IN ["__KGBuilder__", "__Entity__"]][0],
            entity.name,
            type(r),
            [l IN labels(other)
                WHERE NOT l IN ["__KGBuilder__", "__Entity__"]][0],
            other.name
            ]) as values
        RETURN reduce(acc = "", item in values | acc || coalesce(item || ' ', ''))
    } as associated_entities
"""

# Create vector retriever
vector_retriever = VectorCypherRetriever(
    driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    index_name="chunkEmbedding",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

# Create LLM for Text2CypherRetriever
llm = OpenAILLM(
    model_name="gpt-5.2"
)

# Cypher examples as input/query pairs
examples = [
    "USER INPUT: 'Find a node with the name $name?' QUERY: MATCH (node) WHERE toLower(node.name) CONTAINS toLower($name) RETURN node.name AS name, labels(node) AS labels",
]

# Build the retriever
text2cypher_retriever = Text2CypherRetriever(
    driver=driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    llm=llm,
    examples=examples,
)


# tag::memory_settings[]
# Configure agent memory to live in the SAME database as the lesson graph.
# - embedding: matches the lesson graph (text-embedding-3-small, 1536 dims),
#   but memory builds its own vector indexes under its own labels.
# - llm: a cheap model used only for entity/preference extraction + summaries
#   (separate from the agent's gpt-5.2 reasoning model). Change as you like.
# - extraction: use the LLM extractor so we get a rich long-term context graph
#   without needing spaCy/GLiNER/torch.
memory_settings = MemorySettings(
    neo4j={
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": SecretStr(os.getenv("NEO4J_PASSWORD") or ""),
        "database": os.getenv("NEO4J_DATABASE") or "neo4j",
    },
    embedding="openai/text-embedding-3-small",
    llm="openai/gpt-4o-mini",
    extraction={"extractor_type": "llm"},
)
# end::memory_settings[]


# A per-turn reasoning recorder is stashed here so the tools can attach their
# tool calls to the active (:ReasoningTrace) without changing their signatures.
current_recorder: contextvars.ContextVar[StreamingTraceRecorder | None] = (
    contextvars.ContextVar("current_recorder", default=None)
)


async def _record_search(tool_name: str, query: str, result) -> None:
    """Record a graph search into the active reasoning trace (if any)."""
    recorder = current_recorder.get()
    if recorder is None:
        return
    await recorder.record_tool_call(
        tool_name,
        {"query": query},
        result=str(result)[:1000],
        status=ToolCallStatus.SUCCESS,
        auto_observation=True,
    )


# Define the agent's tools. They are async so they can await the reasoning
# recorder; the underlying neo4j-graphrag retrievers are still called directly.

# tag::tools[]
@tool("Get-graph-database-schema")
async def get_schema():
    """Get the schema of the graph database."""
    results, summary, keys = driver.execute_query(
        "CALL db.schema.visualization()",
        database_=os.getenv("NEO4J_DATABASE")
    )
    await _record_search("Get-graph-database-schema", "db.schema.visualization()", results)
    return results


@tool("Search-lesson-content")
async def search_lessons(query: str):
    """Search for lesson content related to the query."""
    result = vector_retriever.search(
        query_text=query,
        top_k=5
    )
    context = [item.content for item in result.items]
    await _record_search("Search-lesson-content", query, context)
    return context


@tool("Query-database")
async def query_database(query: str):
    """A catchall tool to get answers to specific questions about lesson content."""
    result = text2cypher_retriever.get_search_results(query)
    await _record_search("Query-database", query, result)
    return result


tools = [get_schema, search_lessons, query_database]
# end::tools[]

# Create the agent with the model and tools
agent = create_agent(
    model,
    tools
)


# tag::chat_turn[]
async def chat_turn(client: MemoryClient, session_id: str, query: str) -> str:
    """Run one user turn with full short-term / long-term / reasoning memory."""

    # 1. Pull prior context (conversation history + relevant knowledge + similar
    #    past tasks) out of memory and feed it back into the agent.
    context = await client.get_context(query, session_id=session_id)
    system = (
        "You are a helpful assistant for a Neo4j knowledge graph. "
        "Use the tools to search the graph. Relevant memory follows.\n\n" + context
    )

    # 2. Record the human turn (auto-extracts entities/preferences -> long-term).
    await client.short_term.add_message(session_id, "user", query)

    # 3. Run the agent inside a reasoning trace so every graph search is captured.
    final_text = ""
    async with StreamingTraceRecorder(client.reasoning, session_id, task=query) as recorder:
        token = current_recorder.set(recorder)
        try:
            async for step in agent.astream(
                {"messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ]},
                stream_mode="values",
            ):
                message = step["messages"][-1]
                message.pretty_print()
                if getattr(message, "content", None) and message.type == "ai":
                    final_text = message.content
            recorder.set_outcome(final_text[:300] or "completed", success=True)
        finally:
            current_recorder.reset(token)

    # 4. Record the assistant turn.
    if final_text:
        await client.short_term.add_message(session_id, "assistant", final_text)

    return final_text
# end::chat_turn[]


async def main():
    # tag::example_queries[]
    # A mix that exercises all three memory types:
    #  - turn 1 states a preference + mentions domain entities  -> long-term context graph
    #  - turn 2 is a factual lookup                              -> reasoning trace + tool call
    #  - turn 3 relies on remembering turn 1                     -> short-term recall
    queries = [
        "I'm especially interested in RAG and vector search with Neo4j. Which lessons cover those topics?",
        "How many lessons are there in total?",
        "Based on what I told you I'm interested in, which module should I start with?",
    ]
    # end::example_queries[]

    # Reuse a fixed session to accumulate memory across runs, or set SESSION_ID
    # in the environment to start a fresh conversation.
    session_id = os.getenv("SESSION_ID", "workshop-session-001")

    async with MemoryClient(memory_settings) as client:
        for query in queries:
            print("\n" + "=" * 80)
            print(f"USER: {query}")
            print("=" * 80)
            await chat_turn(client, session_id, query)

        # --- Observability: what does the context graph now contain? ---
        print("\n" + "#" * 80)
        print("# MEMORY / CONTEXT GRAPH SUMMARY")
        print("#" * 80)

        stats = await client.get_stats()
        print("\nMemory stats:", stats)

        print("\nHow the graph was searched over time (reasoning traces):")
        traces = await client.reasoning.get_session_traces(session_id)
        for t in sorted(traces, key=lambda x: x.started_at):
            print(f"  [{t.started_at}] task={t.task!r} "
                  f"success={t.success} outcome={t.outcome!r}")

        print("\nTool usage stats:")
        for ts in await client.reasoning.get_tool_stats():
            avg = f"{ts.avg_duration_ms:.0f}ms" if ts.avg_duration_ms is not None else "n/a"
            print(f"  {ts.name}: {ts.total_calls} calls, "
                  f"{ts.success_rate:.0%} success, avg {avg}")

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
