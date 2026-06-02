"""
Lightweight web frontend for the memory-backed Text2Cypher agent.

This is the conventional Python equivalent of jpadams/ford-agent (which is
Spring Boot / Tomcat / JSP): a single-page chat UI with a live graph canvas,
served by **FastAPI** + **uvicorn**, with a vanilla HTML/JS frontend (no build
step). It reuses ``agent_memory.py`` as the backend so every turn flows through
the same short-term / long-term / reasoning memory.

Unlike ford-agent (which visualizes the *queried* graph), the canvas here shows
the **context graph for the current conversation** — messages, the entities they
mention, and the reasoning traces / tool calls (i.e. how the graph was searched).
That is the thing this project is actually about.

Endpoints (mirroring ford-agent):
    GET  /                      -> serve the single-page UI
    POST /chat                  -> {message, conversationId?} -> {conversationId, reply, viz}
    GET  /chat?conversationId=  -> conversation history
    POST /chat/new              -> mint a fresh conversationId
    POST /chat/feedback         -> log 👍/👎 as a preference in the context graph
    GET  /graph/context?conversationId= -> the context-graph viz payload

Run (conventional + lightweight):
    python workshop-genai/memory/agent_memory_web.py
    # then open http://127.0.0.1:8000
"""

import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Reuse the memory-backed agent defined in agent_memory.py. Importing it sets up
# the Neo4j driver, the neo4j-graphrag retrievers, the LangChain agent, the
# memory settings, and the per-turn reasoning recorder — no work is duplicated.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_memory as am  # noqa: E402
from neo4j_agent_memory import MemoryClient  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

# A single long-lived MemoryClient is shared across requests (opened on startup,
# closed on shutdown). connect() also (idempotently) ensures the memory schema.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MemoryClient(am.memory_settings)
    await client.connect()
    state["client"] = client
    try:
        yield
    finally:
        await client.close()
        am.driver.close()


app = FastAPI(title="Agent Memory Chat", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Context-graph visualization
# ---------------------------------------------------------------------------

# Colors keyed by primary node label (Neo4j-browser-ish palette).
_NODE_COLORS = {
    "Conversation": "#F79767",
    "Message": "#57C7E3",
    "Entity": "#8DCC93",
    "Preference": "#D9C8AE",
    "ReasoningTrace": "#ECB5C9",
    "ReasoningStep": "#C990C0",
    "ToolCall": "#4C8EDA",
    "Tool": "#FFC454",
}

# A label-agnostic caption that never returns embedding vectors.
_CAP = (
    "left(coalesce(toString({n}.name), toString({n}.task), toString({n}.tool_name), "
    "toString({n}.action), toString({n}.thought), toString({n}.content), "
    "toString({n}.session_id), labels({n})[0]), 60)"
)


def _branch(match: str) -> str:
    """Build one UNION branch returning the (s)-[r]->(e) triple for the canvas."""
    return f"""
{match}
RETURN elementId(s) AS s_id, labels(s)[0] AS s_label, {_CAP.format(n='s')} AS s_cap,
       elementId(r) AS r_id, type(r) AS r_type,
       elementId(e) AS e_id, labels(e)[0] AS e_label, {_CAP.format(n='e')} AS e_cap
"""


# Every relationship shape the memory layer creates for a session.
_VIZ_QUERY = "\nUNION\n".join(
    _branch(m)
    for m in [
        "MATCH (s:Conversation {session_id:$sid})-[r:HAS_MESSAGE]->(e:Message)",
        "MATCH (:Conversation {session_id:$sid})-[:HAS_MESSAGE]->(s:Message)-[r:MENTIONS]->(e:Entity)",
        "MATCH (:Conversation {session_id:$sid})-[:HAS_MESSAGE]->(s:Message)-[r:NEXT_MESSAGE]->(e:Message)",
        "MATCH (s:ReasoningTrace {session_id:$sid})-[r:HAS_STEP]->(e:ReasoningStep)",
        "MATCH (:ReasoningTrace {session_id:$sid})-[:HAS_STEP]->(s:ReasoningStep)-[r:USES_TOOL]->(e:ToolCall)",
        "MATCH (:ReasoningTrace {session_id:$sid})-[:HAS_STEP]->(:ReasoningStep)"
        "-[:USES_TOOL]->(s:ToolCall)-[r:INSTANCE_OF]->(e:Tool)",
    ]
)


def build_context_viz(session_id: str) -> dict:
    """Return {nodes, relationships} for the session's context subgraph (NVL shape)."""
    records, _, _ = am.driver.execute_query(
        _VIZ_QUERY, sid=session_id, database_=os.getenv("NEO4J_DATABASE")
    )
    nodes: dict[str, dict] = {}
    rels: list[dict] = []
    for row in records:
        for prefix in ("s", "e"):
            nid = row[f"{prefix}_id"]
            label = row[f"{prefix}_label"]
            cap = row[f"{prefix}_cap"]
            nodes[nid] = {
                "id": nid,
                "caption": f"{label}: {cap}" if cap and cap != label else label,
                "label": label,
                "color": _NODE_COLORS.get(label, "#CCCCCC"),
            }
        rels.append({
            "id": row["r_id"],
            "from": row["s_id"],
            "to": row["e_id"],
            "caption": row["r_type"],
        })
    return {"nodes": list(nodes.values()), "relationships": rels}


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversationId: str | None = None


class FeedbackRequest(BaseModel):
    conversationId: str
    rating: str                       # "up" or "down"
    about: str | None = None          # the assistant text the rating refers to


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat/new")
async def chat_new():
    return {"conversationId": f"web-{uuid.uuid4().hex[:12]}"}


@app.post("/chat")
async def chat(req: ChatRequest):
    client: MemoryClient = state["client"]
    session_id = req.conversationId or f"web-{uuid.uuid4().hex[:12]}"
    reply = await am.chat_turn(client, session_id, req.message)
    return {
        "conversationId": session_id,
        "reply": reply,
        "viz": build_context_viz(session_id),
    }


@app.get("/chat")
async def chat_history(conversationId: str):
    client: MemoryClient = state["client"]
    conv = await client.short_term.get_conversation(conversationId, limit=200)
    return {
        "conversationId": conversationId,
        "messages": [{"role": m.role.value, "content": m.content} for m in conv.messages],
    }


@app.get("/graph/context")
async def graph_context(conversationId: str):
    return build_context_viz(conversationId)


@app.post("/chat/feedback")
async def chat_feedback(req: FeedbackRequest):
    """Log a 👍/👎 as a preference, so feedback lives in the context graph too."""
    client: MemoryClient = state["client"]
    await client.long_term.add_preference(
        category="ui_feedback",
        preference=f"User gave a thumbs-{req.rating}",
        context=(req.about or "")[:200],
    )
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
