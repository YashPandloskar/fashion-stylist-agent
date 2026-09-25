"""
Agent tools — each function is one node in the LangGraph graph.
During development: uses Ollama (llama3) + local SDXL.
For final demo: swap USE_OPENAI=true in .env to use GPT-4o + DALL-E 3.
"""

import os
import json
from agent.state import AgentState

MAX_COMPOSE_RETRIES = 2  # so the composer runs at most 3 times per request
from rag.retriever import retrieve_outfit_items
from generation.image_gen import generate_outfit_image


# ── Shared LLM loader ──────────────────────────────────────────────────────────

def _get_llm():
    """Returns either a local Ollama LLM or OpenAI GPT-4o, based on .env flag."""
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"

    if use_openai:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0.7)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3", temperature=0.7)


# ── Tool 1: Context Builder ────────────────────────────────────────────────────

def build_style_brief(state: AgentState) -> AgentState:
    """
    Takes raw user inputs and produces a structured style brief.
    This is the first reasoning step — the LLM interprets context
    (e.g. 'evening + rainy + business dinner') into fashion language.
    """
    llm = _get_llm()

    prompt = f"""You are a professional fashion stylist AI.
    
A customer needs an outfit. Here is their context:
- Occasion: {state.occasion}
- Gender: {state.gender}
- Time of day: {state.time_of_day}
- Weather: {state.weather}
- Style preference: {state.style_preference}

Write a concise style brief (3-5 sentences) describing:
1. The overall aesthetic and mood the outfit should convey
2. Key garment types needed (e.g. outerwear, top, bottom, footwear)
3. Colours, fabrics or textures that suit the weather and occasion
4. Any styling considerations (formality level, layering, accessories)

Be specific and use fashion terminology. Output only the brief, no preamble."""

    response = llm.invoke(prompt)
    state.style_brief = response.content
    return state


# ── Tool 2: Multimodal RAG Retriever ──────────────────────────────────────────

def retrieve_items(state: AgentState) -> AgentState:
    """
    Uses the style brief to query the ChromaDB vector store.
    Returns the most semantically similar fashion items.
    Falls back and sets error if retrieval returns nothing useful.
    """
    if not state.style_brief:
        state.error = "Style brief is empty — cannot retrieve items."
        return state

    items = retrieve_outfit_items(
        query=state.style_brief,
        gender=state.gender,
        n_results=12  # retrieve more than needed; composer will select
    )

    if not items:
        state.error = "No items retrieved from vector store."
        return state

    state.retrieved_items = items
    return state


# ── Tool 3: Outfit Composer ────────────────────────────────────────────────────

def compose_outfit(state: AgentState) -> AgentState:
    """
    Given retrieved items and the style brief, the LLM selects one
    coherent outfit (top, bottom, shoes, optional accessory) using
    structured JSON output for reliable downstream parsing.
    """
    # An error from an earlier tool stops the run; an error left by our own failed
    # attempt (compose_retry_pending) does not, so that we can try again
    if (state.error and not state.compose_retry_pending) or not state.retrieved_items:
        return state
    state.compose_retry_pending = False
    state.error = None

    llm = _get_llm()

    # Format retrieved items for the prompt
    items_text = "\n".join([
        f"[{i+1}] {item['name']} | Category: {item['category']} | "
        f"Colour: {item.get('colour', 'N/A')} | Description: {item.get('description', '')}"
        for i, item in enumerate(state.retrieved_items)
    ])

    prompt = f"""You are a professional fashion stylist AI.

Style brief: {state.style_brief}

Available items:
{items_text}

Select items to form one complete, cohesive outfit. 
You MUST select exactly one item per category where available: top, bottom, shoes.
Optionally add one accessory if it suits the brief.

Respond ONLY with valid JSON in this exact format:
{{
  "top": {{"index": <number>, "name": "<name>", "reason": "<why this works>"}},
  "bottom": {{"index": <number>, "name": "<name>", "reason": "<why this works>"}},
  "shoes": {{"index": <number>, "name": "<name>", "reason": "<why this works>"}},
  "accessory": {{"index": <number>, "name": "<name>", "reason": "<why this works>"}}
}}
The "accessory" entry is optional. Do not add comments or any text outside the JSON."""

    response = llm.invoke(prompt)

    try:
        # Strip any markdown fences if present
        raw = response.content.strip().replace("```json", "").replace("```", "")
        state.outfit_composition = json.loads(raw)
    except json.JSONDecodeError:
        if state.retry_count < MAX_COMPOSE_RETRIES:
            state.retry_count += 1
            state.compose_retry_pending = True
            state.error = "Outfit composer returned malformed JSON. Will retry."
        else:
            state.error = (
                f"Outfit composer returned malformed JSON after {MAX_COMPOSE_RETRIES} retries."
            )

    return state


# ── Tool 4: Image Generator ────────────────────────────────────────────────────

def generate_image(state: AgentState) -> AgentState:
    """
    Constructs a detailed image prompt from the outfit composition
    and passes it to either DALL-E 3 (OpenAI) or SDXL (local).
    """
    if state.error or not state.outfit_composition:
        return state

    outfit = state.outfit_composition
    pieces = [v["name"] for v in outfit.values() if isinstance(v, dict) and "name" in v]
    outfit_description = ", ".join(pieces)

    image_prompt = (
        f"A fashion editorial photograph of a {state.gender} model wearing: {outfit_description}. "
        f"Setting appropriate for {state.occasion} in {state.weather} weather, {state.time_of_day}. "
        f"Style: {state.style_preference}. "
        "Clean studio lighting, high fashion photography, full body shot, white background."
    )

    image_path = generate_outfit_image(image_prompt)
    state.generated_image_path = image_path
    return state


# ── Tool 5: Stylist Narrator ───────────────────────────────────────────────────

def narrate_outfit(state: AgentState) -> AgentState:
    """
    Writes a short, engaging explanation of the outfit choices —
    the kind of copy a stylist or fashion editor would produce.
    This is what the user reads alongside the generated image.
    """
    if state.error or not state.outfit_composition:
        return state

    llm = _get_llm()

    outfit = state.outfit_composition
    pieces_text = "\n".join([
        f"- {slot.capitalize()}: {details['name']} — {details.get('reason', '')}"
        for slot, details in outfit.items()
        if isinstance(details, dict)
    ])

    prompt = f"""You are a fashion stylist writing a short outfit description for a client.

The occasion is: {state.occasion}
The weather is: {state.weather}

The selected outfit:
{pieces_text}

Write 2-3 sentences in a warm, editorial tone explaining why this outfit works for the occasion.
Be specific about the pieces. Do not use generic phrases like 'this look is perfect for'.
Output only the narration."""

    response = llm.invoke(prompt)
    state.stylist_narration = response.content
    return state
