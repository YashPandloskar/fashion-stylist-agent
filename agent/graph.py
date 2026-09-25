"""
LangGraph orchestration for the Fashion Stylist Agent.

Graph structure:
  build_style_brief → retrieve_items → compose_outfit → generate_image → narrate_outfit
                                              ↑ (retry on malformed JSON, max 2 times)
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.tools import (
    build_style_brief,
    retrieve_items,
    compose_outfit,
    generate_image,
    narrate_outfit,
)


# ── Conditional edge: should we retry outfit composition? ──────────────────────

def should_retry_composition(state: AgentState) -> str:
    """
    Loop back to the composer only if it asked for a retry. The composer owns the
    retry bookkeeping (see compose_outfit): LangGraph does not keep changes made to
    the state inside a routing function, so this function must only read it.
    """
    return "retry" if state.compose_retry_pending else "continue"


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes (each node = one tool function)
    graph.add_node("build_style_brief", build_style_brief)
    graph.add_node("retrieve_items", retrieve_items)
    graph.add_node("compose_outfit", compose_outfit)
    graph.add_node("generate_image", generate_image)
    graph.add_node("narrate_outfit", narrate_outfit)

    # Linear edges
    graph.set_entry_point("build_style_brief")
    graph.add_edge("build_style_brief", "retrieve_items")
    graph.add_edge("retrieve_items", "compose_outfit")

    # Conditional edge: retry composition if JSON parsing fails
    graph.add_conditional_edges(
        "compose_outfit",
        should_retry_composition,
        {
            "retry": "compose_outfit",
            "continue": "generate_image",
        }
    )

    graph.add_edge("generate_image", "narrate_outfit")
    graph.add_edge("narrate_outfit", END)

    return graph.compile()


# ── Public runner ──────────────────────────────────────────────────────────────

def run_stylist_agent(
    occasion: str,
    gender: str,
    time_of_day: str,
    weather: str,
    style_preference: str,
) -> AgentState:
    """Entry point — call this from the API or Streamlit app."""

    initial_state = AgentState(
        occasion=occasion,
        gender=gender,
        time_of_day=time_of_day,
        weather=weather,
        style_preference=style_preference,
    )

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return final_state
