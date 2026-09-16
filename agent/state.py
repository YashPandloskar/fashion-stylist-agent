from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentState:
    # --- User inputs ---
    occasion: str = ""
    gender: str = ""
    time_of_day: str = ""
    weather: str = ""
    style_preference: str = ""

    # --- Intermediate outputs (filled in by each tool) ---
    style_brief: str = ""                      # Tool 1: structured style description
    retrieved_items: list = field(default_factory=list)   # Tool 2: items from vector store
    outfit_composition: dict = field(default_factory=dict) # Tool 3: selected outfit pieces
    generated_image_path: Optional[str] = None  # Tool 4: path/URL to generated image
    stylist_narration: str = ""                 # Tool 5: final text explanation

    # --- Control flow ---
    error: Optional[str] = None
    retry_count: int = 0
