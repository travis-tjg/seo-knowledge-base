#!/usr/bin/env python3
"""
MCP Server for SEO Knowledge Base.

Exposes the SEO knowledge base to Claude Desktop, VS Code, and other MCP clients.
"""

import sys
import json
import logging
import os
import io

# Suppress ALL stdout output - MCP uses stdout for JSON protocol
# Redirect any rogue stdout writes to stderr
class StdoutToStderr:
    def write(self, text):
        sys.stderr.write(text)
    def flush(self):
        sys.stderr.flush()

# Save original stdout for MCP, redirect print() calls to stderr
_original_stdout = sys.stdout
sys.stdout = StdoutToStderr()

# Suppress verbose logging from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

# Configure our logging to stderr
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Restore stdout for MCP protocol
sys.stdout = _original_stdout

# Suppress chromadb telemetry message
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Enable quiet mode for query module (no console output)
os.environ["SEO_KB_QUIET"] = "1"

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

from mcp.server.fastmcp import FastMCP
from src.query import query, get_query_engine
from src.embeddings import get_embeddings_manager
from config import DOMAINS

# Initialize MCP server
mcp = FastMCP("knowledge-base")


@mcp.tool()
def query_knowledge(question: str, domain: str = "auto") -> str:
    """
    Query the knowledge base with automatic or explicit domain routing.

    This tool intelligently routes your question to the appropriate knowledge base(s).
    Use domain="auto" to let the system determine the best domain(s) to query.

    Args:
        question: Your question
        domain: Domain selection:
               - "auto" (default): Smart routing based on question content
               - "seo": Query SEO knowledge base only
               - "web_builder": Query AI Website Building knowledge base only
               - "all": Query both domains and combine results

    Returns:
        An answer with citations to source videos and timestamps
    """
    try:
        # Map domain parameter
        actual_domain = None if domain == "auto" else domain
        result = query(question, domain=actual_domain)
        return result
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"


@mcp.tool()
def query_seo_knowledge(question: str) -> str:
    """
    Query the SEO knowledge base for answers about SEO, content marketing,
    local SEO, technical SEO, and digital marketing topics.

    This is a legacy tool - prefer using query_knowledge() with domain="seo".

    Args:
        question: Your question about SEO or digital marketing

    Returns:
        An answer with citations to source videos and timestamps
    """
    try:
        result = query(question, domain="seo")
        return result
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"


@mcp.tool()
def query_web_builder_knowledge(question: str) -> str:
    """
    Query the AI Website Building knowledge base for answers about website builders,
    no-code tools, web design, WordPress, and page builders.

    Args:
        question: Your question about website building or web design

    Returns:
        An answer with citations to source videos and timestamps
    """
    try:
        result = query(question, domain="web_builder")
        return result
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}"


@mcp.tool()
def get_all_stats() -> str:
    """
    Get statistics about all knowledge bases.

    Returns information about each domain including:
    - Total number of chunks
    - Breakdown by source type (YouTube, local files, local videos)
    - Collection name and embedding model
    """
    try:
        all_stats = {}

        for domain_name, domain_config in DOMAINS.items():
            manager = get_embeddings_manager(domain_name)
            stats = manager.get_stats()
            source_stats = manager.get_stats_by_source()

            all_stats[domain_name] = {
                "display_name": domain_config["display_name"],
                "total_chunks": stats['total_chunks'],
                "collection_name": stats['collection_name'],
                "embedding_model": stats['embedding_model'],
                "breakdown": {
                    "youtube_transcripts": source_stats.get('youtube', 0),
                    "local_files": source_stats.get('file', 0),
                    "local_videos": source_stats.get('local_video', 0),
                    "unknown": source_stats.get('unknown', 0)
                }
            }

        return json.dumps(all_stats, indent=2)
    except Exception as e:
        return f"Error getting stats: {str(e)}"


@mcp.tool()
def get_knowledge_base_stats() -> str:
    """
    Get statistics about the SEO knowledge base (legacy).

    Prefer using get_all_stats() to see all knowledge bases.
    """
    try:
        manager = get_embeddings_manager("seo")
        stats = manager.get_stats()
        source_stats = manager.get_stats_by_source()

        result = {
            "total_chunks": stats['total_chunks'],
            "collection_name": stats['collection_name'],
            "embedding_model": stats['embedding_model'],
            "breakdown": {
                "youtube_transcripts": source_stats.get('youtube', 0),
                "local_files": source_stats.get('file', 0),
                "local_videos": source_stats.get('local_video', 0),
                "unknown": source_stats.get('unknown', 0)
            }
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting stats: {str(e)}"


@mcp.tool()
def search_raw_chunks(question: str, top_k: int = 5, domain: str = "seo") -> str:
    """
    Search the knowledge base and return raw chunks without LLM synthesis.

    Useful for seeing exactly what content is retrieved for a query,
    inspecting source material, or debugging retrieval quality.

    Args:
        question: Search query
        top_k: Number of results to return (default 5, max 20)
        domain: Domain to search ("seo", "web_builder", or "all")

    Returns:
        JSON array of matching chunks with metadata (title, channel, timestamp, text)
    """
    try:
        # Clamp top_k
        top_k = max(1, min(20, top_k))

        # Determine domains to search
        if domain == "all":
            domains_to_search = list(DOMAINS.keys())
        elif domain in DOMAINS:
            domains_to_search = [domain]
        else:
            domains_to_search = ["seo"]  # Default

        all_results = []

        for domain_name in domains_to_search:
            manager = get_embeddings_manager(domain_name)

            # Check if we have content
            stats = manager.get_stats()
            if stats['total_chunks'] == 0:
                continue

            # Retrieve chunks
            retriever = manager.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(question)

            for node in nodes:
                meta = node.node.metadata

                # Format timestamp
                start_time = meta.get('start_time', 0)
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                secs = int(start_time % 60)
                if hours > 0:
                    timestamp = f"{hours}:{minutes:02d}:{secs:02d}"
                else:
                    timestamp = f"{minutes}:{secs:02d}"

                all_results.append({
                    "domain": domain_name,
                    "score": round(node.score, 4) if hasattr(node, 'score') else None,
                    "video_title": meta.get('video_title') or meta.get('file_name', 'Unknown'),
                    "channel_name": meta.get('channel_name', 'Unknown'),
                    "timestamp": timestamp,
                    "timestamp_url": meta.get('timestamp_url', ''),
                    "text": node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text
                })

        if not all_results:
            return json.dumps({"error": "No content found in the specified knowledge base(s)"})

        # Sort by score and limit
        all_results.sort(key=lambda x: x['score'] or 0, reverse=True)
        all_results = all_results[:top_k]

        # Add rank after sorting
        for i, result in enumerate(all_results, 1):
            result['rank'] = i

        return json.dumps(all_results, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("kb://stats")
def resource_stats() -> str:
    """Knowledge base statistics overview."""
    return get_all_stats()


@mcp.resource("kb://help")
def resource_help() -> str:
    """Help and usage information for the Knowledge Base."""
    return """
# Multi-Domain Knowledge Base - MCP Server

## Available Tools

### query_knowledge(question, domain="auto")
Query the knowledge base with smart routing. The system automatically
determines which knowledge base(s) to search based on your question.

Domain options:
- "auto" (default): Smart routing based on question content
- "seo": SEO knowledge only
- "web_builder": AI Website Building knowledge only
- "all": Search both and combine results

Example: query_knowledge("What are the best practices for local SEO?")
Example: query_knowledge("Which page builder is best?", domain="web_builder")

### query_seo_knowledge(question)
Query SEO knowledge base directly (legacy tool).

### query_web_builder_knowledge(question)
Query AI Website Building knowledge base directly.

### get_all_stats()
Get statistics about all knowledge bases.

### search_raw_chunks(question, top_k=5, domain="seo")
Search and return raw chunks without LLM synthesis.
Supports domain parameter to search specific knowledge bases.

## Knowledge Bases

### SEO Knowledge Base
- SEO Fight Club streams
- Digitaleer videos
- GBP Roast content
- TestedSEO mastermind recordings

### AI Website Building Knowledge Base
- Website builder tutorials
- No-code tool guides
- Web design best practices

## Tips

- Use domain="auto" for questions that might span multiple topics
- Be specific in your questions for better results
- Use search_raw_chunks to inspect retrieved content
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
