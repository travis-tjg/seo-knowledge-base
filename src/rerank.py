"""Cross-encoder re-ranking for improved retrieval quality."""

import sys
import os
from rich.console import Console

# Use stderr for console output to avoid interfering with MCP protocol
console = Console(stderr=True)

# Lazy-loaded model instance
_reranker = None
_reranker_available = None


def is_reranker_available() -> bool:
    """Check if the reranker can be loaded."""
    global _reranker_available
    if _reranker_available is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_available = True
        except ImportError:
            console.print("[yellow]sentence-transformers not available, skipping re-ranking[/yellow]")
            _reranker_available = False
    return _reranker_available


def get_reranker():
    """Get or initialize the cross-encoder model."""
    global _reranker
    if not is_reranker_available():
        return None
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        console.print("[dim]Loading cross-encoder model...[/dim]")

        # Suppress stdout during model loading to avoid interfering with MCP protocol
        # sentence-transformers prints progress to stdout which breaks JSON-RPC
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
    return _reranker


def rerank_nodes(query: str, nodes: list, top_k: int = None) -> list:
    """
    Re-rank retrieved nodes using a cross-encoder for better relevance.

    Vector search finds semantically similar content, but cross-encoder
    scoring determines how well each chunk actually answers the question.

    Args:
        query: The user's question
        nodes: List of retrieved nodes (from LlamaIndex retriever)
        top_k: Number of top results to return (None = return all, re-ranked)

    Returns:
        Re-ranked list of nodes (best matches first)
    """
    if not nodes:
        return nodes

    reranker = get_reranker()
    if reranker is None:
        # Reranker not available, return nodes as-is
        return nodes

    try:
        # Create query-document pairs for scoring
        pairs = [(query, node.node.text) for node in nodes]

        # Get cross-encoder scores
        scores = reranker.predict(pairs)

        # Attach scores and sort
        for node, score in zip(nodes, scores):
            node.rerank_score = float(score)

        # Sort by cross-encoder score (higher is better)
        nodes.sort(key=lambda n: n.rerank_score, reverse=True)

        if top_k is not None:
            return nodes[:top_k]
        return nodes
    except Exception as e:
        console.print(f"[yellow]Re-ranking failed: {e}[/yellow]")
        return nodes
