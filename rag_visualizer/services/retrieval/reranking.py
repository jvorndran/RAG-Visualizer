"""Reranking module using FlashRank cross-encoders."""

from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_visualizer.services.vector_store import SearchResult

# Lazy import FlashRank
_Ranker: Any | None = None


def _get_ranker() -> Any:
    """Lazy load FlashRank to avoid slow startup."""
    global _Ranker
    if _Ranker is None:
        from flashrank import Ranker

        _Ranker = Ranker
    return _Ranker


def _call_ranker_rerank_compat(
    ranker: Any,
    query: str,
    passages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Handle both old and new FlashRank rerank() signatures."""
    params = list(signature(type(ranker).rerank).parameters)

    if len(params) == 2:
        from flashrank import RerankRequest

        request = RerankRequest(query=query, passages=passages)
        return ranker.rerank(request)

    return ranker.rerank(query, passages)


@dataclass
class RerankerConfig:
    """Configuration for reranking."""

    enabled: bool = False
    model: str = "ms-marco-MiniLM-L-12-v2"
    top_n: int = 5


def rerank_results(
    query: str,
    results: list["SearchResult"],
    config: RerankerConfig,
) -> list["SearchResult"]:
    """Rerank search results using cross-encoder.

    Args:
        query: The search query
        results: List of search results to rerank
        config: Reranking configuration

    Returns:
        Reranked and filtered list of search results
    """
    from rag_visualizer.services.vector_store import SearchResult

    if not config.enabled or not results:
        return results

    Ranker = _get_ranker()
    ranker = Ranker(model_name=config.model)

    # Prepare passages for FlashRank
    passages = [{"id": i, "text": r.text} for i, r in enumerate(results)]

    # Rerank
    reranked = _call_ranker_rerank_compat(ranker, query, passages)

    # Map back to SearchResults with new scores
    reranked_results = []
    for item in reranked[: config.top_n]:
        original = results[item["id"]]
        reranked_results.append(
            SearchResult(
                index=original.index,
                score=item["score"],
                text=original.text,
                metadata={
                    **original.metadata,
                    "original_score": original.score,
                    "reranked": True,
                    "reranker_model": config.model,
                },
            )
        )

    return reranked_results
