"""
FastAPI backend — exposes the Fashion Stylist Agent as a REST API.
Run with: uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import base64

from agent.graph import run_stylist_agent

app = FastAPI(
    title="Fashion Stylist Agent",
    description="Agentic AI outfit recommender using LangGraph, FashionCLIP RAG, and image generation.",
    version="1.0.0",
)

# Serve generated images as static files
Path("outputs").mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ── Request / Response schemas ─────────────────────────────────────────────────

class OutfitRequest(BaseModel):
    occasion: str
    gender: str
    time_of_day: str
    weather: str
    style_preference: str = "smart casual"


class OutfitPiece(BaseModel):
    name: str
    reason: str


class OutfitResponse(BaseModel):
    style_brief: str
    outfit: dict
    stylist_narration: str
    image_path: str | None
    image_base64: str | None  # included for easy Streamlit display
    error: str | None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=OutfitResponse)
def recommend_outfit(request: OutfitRequest):
    """
    Main endpoint — runs the full LangGraph agent pipeline and returns
    the outfit recommendation, narration, and generated image.
    """
    try:
        result = run_stylist_agent(
            occasion=request.occasion,
            gender=request.gender,
            time_of_day=request.time_of_day,
            weather=request.weather,
            style_preference=request.style_preference,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Encode image as base64 for direct Streamlit display
    image_base64 = None
    if result.generated_image_path and Path(result.generated_image_path).exists():
        with open(result.generated_image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return OutfitResponse(
        style_brief=result.style_brief,
        outfit=result.outfit_composition,
        stylist_narration=result.stylist_narration,
        image_path=result.generated_image_path,
        image_base64=image_base64,
        error=result.error,
    )
