"""
ChromaDB vector store for fashion item retrieval.
Run data/ingest_polyvore.py once to populate the store before using the agent.
"""

import chromadb
from chromadb.config import Settings
from rag.embeddings import embed_text
from typing import Optional
import os

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "fashion_items"

# Module-level client cache
_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )
    return _collection


def add_items(items: list[dict]):
    """
    Ingest a list of fashion items into ChromaDB.
    Each item dict should have: id, name, category, colour, description, gender.
    The description is embedded with FashionCLIP for semantic retrieval.
    """
    collection = _get_collection()

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for item in items:
        ids.append(str(item["id"]))
        # Embed a rich text representation for best retrieval quality
        text_to_embed = (
            f"{item['name']}. {item.get('description', '')}. "
            f"Category: {item['category']}. Colour: {item.get('colour', '')}."
        )
        embeddings.append(embed_text(text_to_embed))
        metadatas.append({
            "name": item["name"],
            "category": item["category"],
            "colour": item.get("colour", ""),
            "gender": item.get("gender", "unisex"),
            "description": item.get("description", ""),
        })
        documents.append(text_to_embed)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    print(f"Ingested {len(items)} items into ChromaDB.")


def query_items(
    query_text: str,
    gender: Optional[str] = None,
    n_results: int = 12,
) -> list[dict]:
    """
    Query the vector store with a text description.
    Optionally filter by gender metadata.
    Returns a list of item dicts with name, category, colour, description.
    """
    collection = _get_collection()
    query_embedding = embed_text(query_text)

    where_filter = None
    if gender and gender.lower() not in ("unisex", ""):
        where_filter = {"gender": {"$in": [gender.lower(), "unisex"]}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["metadatas", "distances"],
    )

    items = []
    for metadata in results["metadatas"][0]:
        items.append(metadata)

    return items


def item_count() -> int:
    """Returns the number of items currently in the vector store."""
    return _get_collection().count()
