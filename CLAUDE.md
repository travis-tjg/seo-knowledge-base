# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-domain RAG (Retrieval-Augmented Generation) system for knowledge management. It supports multiple knowledge bases (SEO and AI Website Building) with an intelligent router that can auto-detect which domain(s) to query. Content is ingested from YouTube videos, local documents, and audio/video files, stored as embeddings in ChromaDB, and queried using Claude.

## Common Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the interactive CLI
python src/cli.py

# Run the MCP server (for Claude Desktop/VS Code integration)
python mcp_server.py

# Run the Slack bot
python slack_bot.py

# Batch ingestion utilities
python fast_ingest.py                  # Parallel ingestion with logging
python ingest_all.py                   # Multi-channel batch processor
python ingest_desktop_videos.py        # Local video transcription
python ingest_screamingfrog.py         # Screaming Frog SEO data import
```

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=sk-...          # For embeddings (text-embedding-3-small)
ANTHROPIC_API_KEY=sk-ant-...   # For Claude LLM (claude-sonnet-4)
```

Optional for Slack bot:
```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_SIGNING_SECRET=...
```

## Architecture

### Data Flow

```
Content Source → Extract Text → Chunk (500 tokens, 100 overlap) → Embed → ChromaDB
                                                                              ↓
User Query → Embed → Retrieve (top-5 similar) → Build Context → Claude → Response
```

### Core Modules (`src/`)

| Module | Purpose |
|--------|---------|
| `ingest.py` | YouTube transcript fetching with proxy rotation, chunking, caching |
| `embeddings.py` | ChromaDB integration, singleton manager, CRUD operations |
| `query.py` | RAG query engine, Claude integration, citation formatting |
| `cli.py` | Interactive terminal interface with Rich |
| `file_ingest.py` | PDF, DOCX, TXT, MD, HTML parsing |
| `video_ingest.py` | Local video/audio transcription via Whisper |

### Key Design Patterns

- **Singleton pattern**: `get_embeddings_manager()` and `get_query_engine()` return cached instances
- **Callback-based saving**: Ingestion functions accept `save_callback` for incremental persistence
- **Quiet mode**: Set `SEO_KB_QUIET=1` to suppress console output (used by MCP/Slack)
- **Proxy rotation**: 14 Oxylabs ISP proxies for YouTube reliability (anti-bot detection)
- **Thread-safe writes**: Lock mechanism for concurrent database operations

### Storage Locations

Data is stored locally (outside Google Drive) to avoid sync conflicts:
- ChromaDB: `~/.seo-knowledge-base/chroma_db/`
- Transcript cache: `~/.seo-knowledge-base/transcripts/`

### Configuration (`config.py`)

Key settings:
- `CHUNK_SIZE = 500` tokens per chunk
- `CHUNK_OVERLAP = 100` tokens
- `TOP_K = 5` chunks retrieved per query
- `EMBEDDING_MODEL = "text-embedding-3-small"`
- `LLM_MODEL = "claude-sonnet-4-20250514"`
- `COLLECTION_NAME = "seo_transcripts"`

## CLI Usage

The CLI supports multi-domain queries and ingestion:

**Query Commands:**
- `@seo <question>` → Query SEO knowledge only
- `@web <question>` → Query AI Website Building only
- `@all <question>` → Query both domains
- Any other text → Auto-route to appropriate domain

**Ingestion Commands:**
- `ingest seo <url/path>` → Add content to SEO KB
- `ingest web <url/path>` → Add content to Web Builder KB
- `transcribe seo <path>` → Transcribe video to SEO KB
- `transcribe web <path>` → Transcribe video to Web Builder KB

**Other Commands:**
- `stats` → Show statistics for all knowledge bases
- `clear seo` or `clear web` → Clear a specific KB
- `quit` → Exit

## MCP Server

Exposes these tools to MCP clients:

**Primary Tools:**
1. `query_knowledge(question, domain="auto")` - Smart routing query with citations
   - domain: "auto", "seo", "web_builder", or "all"
2. `get_all_stats()` - Statistics for all knowledge bases
3. `search_raw_chunks(question, top_k, domain)` - Raw chunk retrieval

**Legacy Tools (backward compatible):**
- `query_seo_knowledge(question)` - SEO-only query
- `query_web_builder_knowledge(question)` - Web builder-only query
- `get_knowledge_base_stats()` - SEO stats only

## Slack Bot Commands

- `/ask <question>` - Auto-route to appropriate domain
- `/ask-seo <question>` - Query SEO knowledge only
- `/ask-web <question>` - Query Web Builder knowledge only
- `/kb-stats` - Show all knowledge base statistics
- `/seo-stats` - Show SEO stats (legacy)
- `/ingest <url>` - Add content to SEO KB

## Domains Configuration

Defined in `config.py`:
```python
DOMAINS = {
    "seo": {
        "collection_name": "seo_transcripts",
        "display_name": "SEO Knowledge Base",
        "description": "SEO, local SEO, rankings, backlinks, technical SEO",
    },
    "web_builder": {
        "collection_name": "ai_website_builder",
        "display_name": "AI Website Building Knowledge Base",
        "description": "AI website builders, no-code tools, web design",
    },
}
```

## Supported File Formats

Documents: `.pdf`, `.docx`, `.txt`, `.md`, `.html`
Video/Audio: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.mp3`, `.wav`, `.m4a`
