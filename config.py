"""Configuration settings for SEO Knowledge Base RAG System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Paths - Store data locally (outside Google Drive) to avoid sync conflicts
BASE_DIR = Path(__file__).parent
LOCAL_DATA_DIR = Path.home() / ".seo-knowledge-base"  # Local storage for DB
CHROMA_DIR = LOCAL_DATA_DIR / "chroma_db"
TRANSCRIPTS_DIR = LOCAL_DATA_DIR / "transcripts"

# Ensure directories exist
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking settings
CHUNK_SIZE = 500  # Target tokens per chunk
CHUNK_OVERLAP = 100  # Overlap tokens between chunks

# Retrieval settings
TOP_K = 8  # Number of chunks to retrieve (increased for better TestedSEO coverage)

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "claude-sonnet-4-20250514"

# ChromaDB collection name (legacy - use DOMAINS for multi-domain support)
COLLECTION_NAME = "seo_transcripts"

# System prompts for each domain
SYSTEM_PROMPT_SEO = """You are a friendly SEO expert who has studied hundreds of hours of SEO content from
top practitioners. You're chatting with a colleague who wants your take on SEO topics.

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

SYSTEM_PROMPT_WEB_BUILDER = """You are a friendly AI website building expert who has studied hundreds of hours
of content about website builders, no-code tools, and web design best practices. You're chatting with a
colleague who wants your take on building websites.

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

# Domain configurations for multi-domain knowledge base
DOMAINS = {
    "seo": {
        "collection_name": "seo_transcripts",
        "display_name": "SEO Knowledge Base",
        "description": "SEO, local SEO, rankings, backlinks, technical SEO, content marketing",
        "system_prompt": SYSTEM_PROMPT_SEO,
    },
    "web_builder": {
        "collection_name": "ai_website_builder",
        "display_name": "AI Website Building Knowledge Base",
        "description": "AI website builders, no-code tools, web design, WordPress, page builders",
        "system_prompt": SYSTEM_PROMPT_WEB_BUILDER,
    },
}

DEFAULT_DOMAIN = "seo"
