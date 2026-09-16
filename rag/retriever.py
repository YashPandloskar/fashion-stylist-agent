"""
Retriever — called by Tool 2 in the agent.
Thin wrapper around vector_store.query_items.
"""

from rag.vector_store import query_items
from typing import Optional


def retrieve_outfit_items(
    query: str,
    gender: Optional[str] = None,
    n_results: int = 12,
) -> list[dict]:
    """
    Retrieve the most semantically similar fashion items for a given style brief.
    Returns a list of item dicts ready for the outfit composer.
    """
    return query_items(
        query_text=query,
        gender=gender,
        n_results=n_results,
    )
