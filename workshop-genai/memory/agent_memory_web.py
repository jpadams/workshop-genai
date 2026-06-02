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

import colorsys
import hashlib

# Fixed colors for the memory labels (Neo4j-browser-ish palette). Lesson-graph
# labels get a stable color derived from the label name (see _color_for).
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

# Labels the KG builder adds to every node; we hide them when picking a caption.
_HIDDEN_LABELS = "['__KGBuilder__', '__Entity__', '__Node__']"


def _lbl(v: str) -> str:
    """Cypher fragment: the first meaningful label of node variable ``v``."""
    return f"head([l IN labels({v}) WHERE NOT l IN {_HIDDEN_LABELS}] + labels({v}) + ['Node'])"


def _cap(v: str) -> str:
    """Cypher fragment: a short caption for node ``v`` that never ships embeddings."""
    fields = ["name", "title", "url", "task", "tool_name", "action",
              "thought", "content", "text", "session_id"]
    inner = ", ".join(f"toString({v}.{f})" for f in fields)
    return f"left(coalesce({inner}, {_lbl(v)}), 60)"


def _color_for(label: str) -> str:
    """Stable color per label: fixed for memory labels, hashed hue otherwise."""
    if label in _NODE_COLORS:
        return _NODE_COLORS[label]
    hue = int(hashlib.md5(label.encode()).hexdigest(), 16) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.62, 0.55)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def _branch(match: str) -> str:
    """Build one UNION branch returning the (s)-[r]->(e) triple for the canvas."""
    return f"""
{match}
RETURN elementId(s) AS s_id, {_lbl('s')} AS s_label, {_cap('s')} AS s_cap,
       elementId(r) AS r_id, type(r) AS r_type,
       elementId(e) AS e_id, {_lbl('e')} AS e_label, {_cap('e')} AS e_cap
"""


# Every relationship shape that makes up a session's context graph — including
# the RETRIEVED edge from a ToolCall to the lesson-graph nodes it fetched.
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
        "MATCH (:ReasoningTrace {session_id:$sid})-[:HAS_STEP]->(:ReasoningStep)"
        "-[:USES_TOOL]->(s:ToolCall)-[r:RETRIEVED]->(e)",
    ]
)

# Direction-agnostic 1-hop neighborhood of a node, for double-click expand.
_EXPAND_QUERY = f"""
MATCH (n) WHERE elementId(n) = $id
MATCH (n)-[r]-(m)
WITH n, r, m LIMIT 60
RETURN elementId(n) AS n_id, {_lbl('n')} AS n_label, {_cap('n')} AS n_cap,
       elementId(r) AS r_id, type(r) AS r_type, startNode(r) = n AS n_is_start,
       elementId(m) AS m_id, {_lbl('m')} AS m_label, {_cap('m')} AS m_cap
"""


def _node(nid: str, label: str, cap: str) -> dict:
    return {
        "id": nid,
        "caption": f"{label}: {cap}" if cap and cap != label else (label or "Node"),
        "label": label or "Node",
        "color": _color_for(label or "Node"),
    }


def build_context_viz(session_id: str) -> dict:
    """Return {nodes, relationships} for the session's context subgraph (NVL shape)."""
    records, _, _ = am.driver.execute_query(
        _VIZ_QUERY, sid=session_id, database_=os.getenv("NEO4J_DATABASE")
    )
    nodes: dict[str, dict] = {}
    rels: dict[str, dict] = {}
    for row in records:
        nodes[row["s_id"]] = _node(row["s_id"], row["s_label"], row["s_cap"])
        nodes[row["e_id"]] = _node(row["e_id"], row["e_label"], row["e_cap"])
        rels[row["r_id"]] = {
            "id": row["r_id"], "from": row["s_id"], "to": row["e_id"], "caption": row["r_type"],
        }
    return {"nodes": list(nodes.values()), "relationships": list(rels.values())}


def expand_node_viz(element_id: str) -> dict:
    """Return the 1-hop neighborhood of a node (any label) in NVL shape."""
    records, _, _ = am.driver.execute_query(
        _EXPAND_QUERY, id=element_id, database_=os.getenv("NEO4J_DATABASE")
    )
    nodes: dict[str, dict] = {}
    rels: dict[str, dict] = {}
    for row in records:
        nodes[row["n_id"]] = _node(row["n_id"], row["n_label"], row["n_cap"])
        nodes[row["m_id"]] = _node(row["m_id"], row["m_label"], row["m_cap"])
        frm, to = (row["n_id"], row["m_id"]) if row["n_is_start"] else (row["m_id"], row["n_id"])
        rels[row["r_id"]] = {"id": row["r_id"], "from": frm, "to": to, "caption": row["r_type"]}
    return {"nodes": list(nodes.values()), "relationships": list(rels.values())}


def node_details(element_id: str) -> dict:
    """Return a node's label + display-safe properties (no embeddings/huge text)."""
    records, _, _ = am.driver.execute_query(
        f"MATCH (n) WHERE elementId(n) = $id "
        f"RETURN {_lbl('n')} AS label, labels(n) AS labels, properties(n) AS props",
        id=element_id, database_=os.getenv("NEO4J_DATABASE"),
    )
    if not records:
        return {"id": element_id, "label": None, "labels": [], "properties": {}}
    row = records[0]
    props = {}
    for k, v in (row["props"] or {}).items():
        if k == "embedding" or (isinstance(v, list) and v and isinstance(v[0], (int, float))):
            continue  # skip embedding vectors
        props[k] = (v[:500] + "…") if isinstance(v, str) and len(v) > 500 else v
    return {"id": element_id, "label": row["label"], "labels": row["labels"], "properties": props}


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


@app.get("/graph/expand")
async def graph_expand(id: str):
    """Neighbors of a node (memory or lesson-graph) for double-click expand."""
    return expand_node_viz(id)


@app.get("/graph/node")
async def graph_node(id: str):
    """A node's properties for the click-for-info panel."""
    return node_details(id)


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
