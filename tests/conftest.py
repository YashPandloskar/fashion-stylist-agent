"""Test set-up: the retrieval and image-generation modules pull in torch, ChromaDB and diffusers, so they are
replaced with light fakes. The tests then exercise the agent's own logic (the graph and its retry loop) quickly,
without a GPU, a model download or an LLM."""

import sys
import types

import pytest

ITEMS = [
    {"name": "White linen shirt", "category": "top", "colour": "white", "description": ""},
    {"name": "Navy jeans", "category": "bottom", "colour": "navy", "description": ""},
    {"name": "White sneakers", "category": "shoes", "colour": "white", "description": ""},
]

GOOD_JSON = (
    '{"top": {"index": 1, "name": "White linen shirt", "reason": "breathable"},'
    ' "bottom": {"index": 2, "name": "Navy jeans", "reason": "smart casual"},'
    ' "shoes": {"index": 3, "name": "White sneakers", "reason": "clean finish"}}'
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Answers the composer prompt from a script, and every other prompt with plain text."""

    def __init__(self, composer_replies):
        self.composer_replies = list(composer_replies)
        self.composer_calls = 0

    def invoke(self, prompt):
        if "Respond ONLY with valid JSON" in prompt:
            self.composer_calls += 1
            index = min(self.composer_calls, len(self.composer_replies)) - 1
            return FakeResponse(self.composer_replies[index])
        return FakeResponse("A short stylist text.")


@pytest.fixture
def stub_heavy_modules(monkeypatch):
    retriever = types.ModuleType("rag.retriever")
    retriever.items = list(ITEMS)
    retriever.retrieve_outfit_items = lambda query, gender=None, n_results=12: list(retriever.items)
    image_gen = types.ModuleType("generation.image_gen")
    image_gen.generate_outfit_image = lambda prompt: "generated.png"
    for name, module in (
        ("rag", types.ModuleType("rag")),
        ("rag.retriever", retriever),
        ("generation", types.ModuleType("generation")),
        ("generation.image_gen", image_gen),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("agent.tools", "agent.graph"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return retriever


@pytest.fixture
def run_agent(stub_heavy_modules, monkeypatch):
    """Runs the whole graph with a scripted composer; returns (final state, fake llm)."""

    def _run(composer_replies):
        import agent.tools as tools
        from agent.graph import run_stylist_agent

        llm = FakeLLM(composer_replies)
        monkeypatch.setattr(tools, "_get_llm", lambda: llm)
        state = run_stylist_agent(
            occasion="weekend lunch", gender="men", time_of_day="afternoon", weather="sunny", style_preference="casual"
        )
        return state, llm

    return _run
