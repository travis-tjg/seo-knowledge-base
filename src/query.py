"""Query engine for retrieval and Claude interaction."""

import os
import re
from anthropic import Anthropic
from rich.console import Console
from rich.markdown import Markdown

import sys
sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
from config import ANTHROPIC_API_KEY, LLM_MODEL, TOP_K, DOMAINS, DEFAULT_DOMAIN
from src.embeddings import get_embeddings_manager
from src.rerank import rerank_nodes
from src.router import get_router

# Only print status messages in CLI mode (not MCP/Slack)
_quiet_mode = os.environ.get("SEO_KB_QUIET", "0") == "1"
# Use stderr for console output to avoid interfering with MCP protocol
console = Console(stderr=True)

# Legacy system prompt (kept for backward compatibility)
# New code should use DOMAINS[domain]['system_prompt'] from config.py
SYSTEM_PROMPT = DOMAINS[DEFAULT_DOMAIN]['system_prompt']

# Combined system prompt for multi-domain queries
SYSTEM_PROMPT_COMBINED = """You are a knowledgeable expert in both SEO and AI website building who has studied
hundreds of hours of content from top practitioners in both fields. You're chatting with a colleague who
wants your take on topics that may span SEO, web design, and website building.

Your style:
- Be conversational and direct, like you're talking to a coworker
- Share insights naturally, as if recalling what you've learned
- NEVER say things like "Based on the transcript" or "According to the excerpts" or "The provided content shows"
- Instead, speak as if this knowledge is yours: "So the key thing here is..." or "What works really well is..."
- Use citations [1], [2], etc. to reference sources, but do NOT mention people by name unless absolutely necessary
- Use neutral phrasing like "it's believed that..." or "the research suggests..." or "one effective approach is..."
- Be practical and actionable
- Keep it concise but helpful

If the context doesn't have relevant info, just say you're not sure about that specific topic."""


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def build_context(nodes: list) -> tuple[str, list[dict]]:
    """
    Build context string from retrieved nodes.
    Returns context string and list of source metadata for citations.
    """
    context_parts = []
    sources = []

    for i, node in enumerate(nodes, 1):
        meta = node.node.metadata
        timestamp = format_timestamp(meta.get('start_time', 0))

        # Handle different metadata schemas (YouTube vs local videos)
        video_title = meta.get('video_title') or meta.get('file_name') or 'Unknown'
        channel_name = meta.get('channel_name') or meta.get('video_series') or 'Unknown'
        timestamp_url = meta.get('timestamp_url') or meta.get('file_path') or ''

        context_parts.append(
            f"[{i}] **{video_title}** by {channel_name} (at {timestamp})\n"
            f"{node.node.text}\n"
        )

        sources.append({
            'index': i,
            'video_title': video_title,
            'channel_name': channel_name,
            'timestamp': timestamp,
            'timestamp_url': timestamp_url,
        })

    return "\n".join(context_parts), sources


def format_sources(sources: list[dict], response_text: str) -> str:
    """Format sources section with clickable links for cited sources."""
    # Find which sources were actually cited
    cited_indices = set()
    for match in re.finditer(r'\[(\d+)\]', response_text):
        cited_indices.add(int(match.group(1)))

    if not cited_indices:
        return ""

    source_lines = ["\n\n---\n**Sources:**"]
    for source in sources:
        if source['index'] in cited_indices:
            source_lines.append(
                f"[{source['index']}] [{source['video_title']}]({source['timestamp_url']}) "
                f"by {source['channel_name']} @ {source['timestamp']}"
            )

    return "\n".join(source_lines)


class QueryEngine:
    """Handles queries against the knowledge base with multi-domain support."""

    def __init__(self):
        """Initialize the query engine."""
        self.anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.router = get_router()

    def _get_system_prompt(self, domains_queried: list[str]) -> str:
        """Get the appropriate system prompt based on domains queried."""
        if len(domains_queried) == 1:
            # Single domain - use domain-specific prompt
            domain = domains_queried[0]
            return DOMAINS.get(domain, DOMAINS[DEFAULT_DOMAIN])['system_prompt']
        else:
            # Multiple domains - use combined prompt
            return SYSTEM_PROMPT_COMBINED

    def query(
        self,
        question: str,
        conversation_history: list[dict] = None,
        domain: str = None
    ) -> str:
        """
        Query the knowledge base and get an answer from Claude.
        Returns formatted response with citations.

        Args:
            question: The user's question
            conversation_history: Optional list of previous messages in format:
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            domain: Optional domain to query ('seo', 'web_builder', 'all', or None for auto-routing)
        """
        # Determine which domains to query
        domains_to_query = self.router.route(question, domain)

        if not _quiet_mode:
            if len(domains_to_query) > 1:
                console.print(f"[dim]Querying domains: {', '.join(domains_to_query)}...[/dim]")
            else:
                console.print(f"[dim]Querying {domains_to_query[0]} knowledge base...[/dim]")

        # Collect nodes from each domain
        all_nodes = []
        domains_with_content = []

        for domain_name in domains_to_query:
            manager = get_embeddings_manager(domain_name)
            stats = manager.get_stats()

            if stats['total_chunks'] == 0:
                continue

            domains_with_content.append(domain_name)

            # For follow-up questions, combine with previous context for better retrieval
            search_query = question
            if conversation_history:
                recent_context = []
                for msg in conversation_history[-4:]:  # Last 2 exchanges
                    if msg["role"] == "user":
                        recent_context.append(msg["content"])
                if recent_context:
                    search_query = f"{' '.join(recent_context)} {question}"

            # Retrieve relevant chunks from this domain
            retriever = manager.index.as_retriever(similarity_top_k=TOP_K)
            nodes = retriever.retrieve(search_query)

            # Tag nodes with domain for source attribution
            for node in nodes:
                node.node.metadata['_domain'] = domain_name

            all_nodes.extend(nodes)

        if not all_nodes:
            if len(domains_to_query) > 1:
                return "No relevant content found in any knowledge base."
            else:
                return f"The {domains_to_query[0]} knowledge base is empty. Please ingest some content first."

        # Apply boosting to retrieval scores
        SOURCE_BOOST = {'local_video': 1.15}  # 15% boost for local video content

        for node in all_nodes:
            source_type = node.node.metadata.get('source_type', 'unknown')
            source_boost = SOURCE_BOOST.get(source_type, 1.0)
            stored_boost = node.node.metadata.get('boost', 1.0)
            total_boost = source_boost * stored_boost
            if hasattr(node, 'score') and node.score is not None:
                node.score = node.score * total_boost

        # Re-sort by boosted score
        all_nodes.sort(key=lambda n: n.score if hasattr(n, 'score') and n.score else 0, reverse=True)

        # Re-rank using cross-encoder
        if not _quiet_mode:
            console.print(f"[dim]Re-ranking results...[/dim]")
        all_nodes = rerank_nodes(question, all_nodes)

        # Take top results across all domains
        top_nodes = all_nodes[:TOP_K]

        # Build context and get sources
        context, sources = build_context(top_nodes)

        # Query Claude with appropriate system prompt
        if not _quiet_mode:
            console.print(f"[dim]Generating response...[/dim]")

        system_prompt = self._get_system_prompt(domains_with_content)

        messages = []
        if conversation_history:
            messages.extend(conversation_history)

        messages.append({
            "role": "user",
            "content": f"Question: {question}\n\nHere's what the experts have said:\n\n{context}"
        })

        message = self.anthropic.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )

        response_text = message.content[0].text
        sources_section = format_sources(sources, response_text)

        return response_text + sources_section


# Singleton instance
_engine: QueryEngine | None = None


def get_query_engine() -> QueryEngine:
    """Get the singleton query engine instance."""
    global _engine
    if _engine is None:
        _engine = QueryEngine()
    return _engine


def query(question: str, conversation_history: list[dict] = None, domain: str = None) -> str:
    """
    Convenience function to query the knowledge base.

    Args:
        question: The user's question
        conversation_history: Optional list of previous messages
        domain: Optional domain to query ('seo', 'web_builder', 'all', or None for auto-routing)
    """
    engine = get_query_engine()
    return engine.query(question, conversation_history, domain)
