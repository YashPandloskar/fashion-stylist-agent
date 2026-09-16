"""
Polyvore dataset ingestion — run this ONCE before starting the agent.

Downloads the Polyvore-U dataset from HuggingFace, extracts item metadata,
and ingests it into ChromaDB with FashionCLIP embeddings.

Usage:
    python data/ingest_polyvore.py

Note: First run will download the FashionCLIP model (~400MB) and the
Polyvore dataset. Subsequent runs are fast as ChromaDB persists to disk.
"""

import json
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from rag.vector_store import add_items, item_count

BATCH_SIZE = 64          # embed and upsert in batches to manage memory
MAX_ITEMS = 5000         # cap for development; remove for full dataset


def load_polyvore_items() -> list[dict]:
    """
    Loads the Polyvore-U dataset from HuggingFace.
    Falls back to a small synthetic sample if the dataset is unavailable.
    """
    print("Loading Polyvore dataset from HuggingFace...")
    try:
        # Polyvore-U is a well-known fashion compatibility dataset
        dataset = load_dataset("Marqo/polyvore", split="train")
        items = []
        for i, row in enumerate(dataset):
            if i >= MAX_ITEMS:
                break
            items.append({
                "id": str(i),
                "name": row.get("name", f"Item {i}"),
                "category": row.get("category", "clothing"),
                "colour": row.get("colour", ""),
                "description": row.get("description", ""),
                "gender": row.get("gender", "unisex"),
            })
        print(f"Loaded {len(items)} items from Polyvore dataset.")
        return items

    except Exception as e:
        print(f"Could not load Polyvore dataset ({e}).")
        print("Falling back to synthetic sample data for development...")
        return _synthetic_sample()


def _synthetic_sample() -> list[dict]:
    """
    Small hand-crafted dataset for local development and testing
    without needing the full Polyvore download.
    """
    return [
        {"id": "1", "name": "White Oxford Shirt", "category": "top", "colour": "white", "gender": "men", "description": "Classic slim-fit Oxford shirt in white cotton"},
        {"id": "2", "name": "Navy Chinos", "category": "bottom", "colour": "navy", "gender": "men", "description": "Tailored chino trousers in navy blue"},
        {"id": "3", "name": "White Linen Shirt", "category": "top", "colour": "white", "gender": "men", "description": "Breathable linen shirt for warm weather"},
        {"id": "4", "name": "Beige Trench Coat", "category": "outerwear", "colour": "beige", "gender": "women", "description": "Classic double-breasted trench coat"},
        {"id": "5", "name": "Black Midi Skirt", "category": "bottom", "colour": "black", "gender": "women", "description": "Flowy satin midi skirt"},
        {"id": "6", "name": "Cream Silk Blouse", "category": "top", "colour": "cream", "gender": "women", "description": "Elegant silk blouse with relaxed fit"},
        {"id": "7", "name": "White Sneakers", "category": "shoes", "colour": "white", "gender": "unisex", "description": "Minimalist leather sneakers"},
        {"id": "8", "name": "Black Leather Loafers", "category": "shoes", "colour": "black", "gender": "unisex", "description": "Polished penny loafers in smooth leather"},
        {"id": "9", "name": "Camel Wool Coat", "category": "outerwear", "colour": "camel", "gender": "unisex", "description": "Mid-length wool coat for cold weather"},
        {"id": "10", "name": "Tan Leather Tote", "category": "accessory", "colour": "tan", "gender": "women", "description": "Structured leather tote bag"},
        {"id": "11", "name": "Navy Wool Blazer", "category": "top", "colour": "navy", "gender": "men", "description": "Single-breasted wool blazer"},
        {"id": "12", "name": "Grey Slim Trousers", "category": "bottom", "colour": "grey", "gender": "men", "description": "Slim-fit tailored trousers in grey wool"},
        {"id": "13", "name": "Burgundy Knit Jumper", "category": "top", "colour": "burgundy", "gender": "unisex", "description": "Ribbed crew-neck knit in burgundy"},
        {"id": "14", "name": "Dark Wash Jeans", "category": "bottom", "colour": "dark blue", "gender": "unisex", "description": "Straight-leg jeans in dark indigo wash"},
        {"id": "15", "name": "Brown Chelsea Boots", "category": "shoes", "colour": "brown", "gender": "unisex", "description": "Elasticated side-panel Chelsea boots in leather"},
        {"id": "16", "name": "Black Turtleneck", "category": "top", "colour": "black", "gender": "unisex", "description": "Fine-knit wool turtleneck sweater"},
        {"id": "17", "name": "Floral Summer Dress", "category": "dress", "colour": "multicolour", "gender": "women", "description": "Lightweight floral print wrap dress"},
        {"id": "18", "name": "White Linen Trousers", "category": "bottom", "colour": "white", "gender": "unisex", "description": "Relaxed-fit linen trousers for warm weather"},
        {"id": "19", "name": "Denim Jacket", "category": "outerwear", "colour": "light blue", "gender": "unisex", "description": "Classic washed denim jacket"},
        {"id": "20", "name": "Block-heel Sandals", "category": "shoes", "colour": "tan", "gender": "women", "description": "Open-toe block heel sandals in tan suede"},
    ]


def ingest_in_batches(items: list[dict]):
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        print(f"Ingesting batch {i // BATCH_SIZE + 1} ({len(batch)} items)...")
        add_items(batch)
    print(f"\nIngestion complete. Total items in store: {item_count()}")


if __name__ == "__main__":
    items = load_polyvore_items()
    ingest_in_batches(items)
