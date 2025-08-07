# graph.py  ──────────────────────────────────────────────────────────────
from __future__ import annotations

import shutil, textwrap
from pathlib import Path
from typing import List, TypedDict

from IPython.display import Image, display
from langgraph.graph import StateGraph, END

# ────────────────────────────────────────────────────────────────────────
# 0. Assets
# ────────────────────────────────────────────────────────────────────────
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_ASSETS_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# 1. Graph state
# ────────────────────────────────────────────────────────────────────────
class State(TypedDict):
    user_input: str
    messages:   List[str]

# ────────────────────────────────────────────────────────────────────────
# 2. Lazy  
# ────────────────────────────────────────────────────────────────────────
from aimedorchestraagent import aimedorchestraAgent
_ROUTER: aimedorchestraAgent | None = None
def _get_router() -> aimedorchestraAgent:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = aimedorchestraAgent()
    return _ROUTER

# ────────────────────────────────────────────────────────────────────────
# 3. Node factories
# ────────────────────────────────────────────────────────────────────────
def entry_node(state: State):
    return {"messages": state["messages"] + ["🔷 aimedorchestra Router online."]}

def make_agent_node(bucket_key: str, label_wrapped: str):
    def _fn(state: State):
        reply = _get_router().route(state["user_input"])
        return {"messages": state["messages"] + [f"**{label_wrapped}**: {reply}"]}
    _fn.__name__ = f"{bucket_key}_node"
    return _fn

# ────────────────────────────────────────────────────────────────────────
# 4. Labels (plain)  → wrapped at runtime
# ────────────────────────────────────────────────────────────────────────
LABELS = {
    "trial":        "Trial Matcher",
    "synthesis":    "Data Synthesis",
    "diagnostic":   "Diagnostics",
    "diet":         "Diet Planner",
    "drug":         "Drug Discovery",
    "ethical":      "Ethics & Bias",
    "genomic":      "Genomics",
    "image":        "Imaging",
    "literature":   "Literature Surveillance",
    "mental":       "Mental Health",
    "treatment":    "Treatment Optim.",
    "orchestrator": "Orchestrator LLM",
}
WRAPPED = {k: v.replace(" ", "\n") for k, v in LABELS.items()}  # real newline char!

# ────────────────────────────────────────────────────────────────────────
# 5. Router decision
# ────────────────────────────────────────────────────────────────────────
def decide_route(state: State) -> str:
    txt = state["user_input"].lower()
    for kw in LABELS:
        if kw in txt:
            return WRAPPED[kw]
    return WRAPPED["orchestrator"]

# ────────────────────────────────────────────────────────────────────────
# 6. Build LangGraph (no invisible chain needed—radial layout)
# ────────────────────────────────────────────────────────────────────────
def build_router_graph():
    g = StateGraph(State)
    g.add_node("Entry", entry_node)
    g.set_entry_point("Entry")

    for kw, wrapped in WRAPPED.items():
        g.add_node(wrapped, make_agent_node(kw, wrapped))
        g.add_edge(wrapped, END)

    g.add_conditional_edges(
        "Entry", decide_route,
        {w: w for w in WRAPPED.values()},
    )
    return g.compile()

# ────────────────────────────────────────────────────────────────────────
# 7. Render + save PNG
# ────────────────────────────────────────────────────────────────────────
def show_mermaid_png(save_as: str | None = None):
    wf = build_router_graph()
    dg = wf.get_graph()

    # Apply radial / “twopi” layout + spacing
    try:    dg.attr(layout="twopi", nodesep="0.3", ranksep="1")
    except Exception:
        try: dg.graph_attr.update(layout="twopi", nodesep="0.3", ranksep="1")
        except Exception: pass

    # Fallback: print Mermaid if dot not found
    if shutil.which("dot") is None:
        print("⚠️ Graphviz not found. Mermaid code:")
        print(textwrap.indent(dg.draw_mermaid(), "    "))
        return

    png_obj   = dg.draw_mermaid_png()
    png_bytes = png_obj if isinstance(png_obj, (bytes, bytearray)) else Path(png_obj).read_bytes()

    try: display(Image(png_bytes))
    except Exception: pass

    out_path = _ASSETS_DIR / (save_as or "router.png")
    out_path.write_bytes(png_bytes)
    print(f"✅ Router graph saved to {out_path.relative_to(Path.cwd())}")

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_router_graph()
    print("Graph compiled. Run show_mermaid_png() to render & save.")
