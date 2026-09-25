from tests.conftest import GOOD_JSON


def test_happy_path_runs_every_tool_once(run_agent):
    state, llm = run_agent([GOOD_JSON])
    assert llm.composer_calls == 1
    assert state["error"] is None
    assert state["generated_image_path"] == "generated.png"
    assert state["stylist_narration"]
    assert set(state["outfit_composition"]) == {"top", "bottom", "shoes"}


def test_json_wrapped_in_markdown_fences_is_accepted(run_agent):
    state, llm = run_agent(["```json\n" + GOOD_JSON + "\n```"])
    assert llm.composer_calls == 1
    assert state["error"] is None


def test_one_malformed_reply_is_retried_and_recovers(run_agent):
    state, llm = run_agent(["this is not json", GOOD_JSON])
    assert llm.composer_calls == 2
    assert state["retry_count"] == 1
    assert state["error"] is None
    assert state["generated_image_path"] == "generated.png"


def test_retries_stop_after_two_and_the_run_ends_cleanly(run_agent):
    # the old routing function never stopped retrying; this must now finish after 1 attempt + 2 retries
    state, llm = run_agent(["not json"] * 10)
    assert llm.composer_calls == 3
    assert state["retry_count"] == 2
    assert "malformed JSON" in state["error"]
    assert state["generated_image_path"] is None
    assert state["stylist_narration"] == ""


def test_nothing_retrieved_stops_before_the_composer(run_agent, stub_heavy_modules):
    stub_heavy_modules.items = []
    state, llm = run_agent([GOOD_JSON])
    assert llm.composer_calls == 0
    assert state["error"] == "No items retrieved from vector store."
    assert state["generated_image_path"] is None
