#!/usr/bin/env python3
"""
Slack Bot for SEO Knowledge Base.

Provides access to the SEO knowledge base via:
- @mentions: @SEOBot what is local SEO?
- Slash command: /ask what is local SEO?
- URL ingestion: /ingest <url> or just paste URLs to ingest content

Uses Socket Mode (no public URL required).
"""

import os
import re
import sys
import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Question logging
QUESTIONS_LOG_FILE = Path(__file__).parent / "data" / "slack_questions.json"
QUESTIONS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from src.query import query
from src.embeddings import get_embeddings_manager
from config import DOMAINS
from src.ingest import (
    is_youtube_url,
    extract_video_id,
    get_video_info,
    get_transcript,
    chunk_transcript,
    save_transcript,
)

# Initialize Slack app
app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Store bot user ID for mention parsing
BOT_USER_ID = None

# Conversation memory store
# Key: thread_ts or channel_id (for DMs without threads)
# Value: list of {"role": "user"|"assistant", "content": str}
CONVERSATION_MEMORY: dict[str, list[dict]] = {}
MAX_HISTORY_LENGTH = 10  # Keep last 10 messages (5 exchanges)
MEMORY_EXPIRY_SECONDS = 3600  # Clear conversations after 1 hour of inactivity
MEMORY_TIMESTAMPS: dict[str, float] = {}  # Track last activity time


def get_conversation_key(channel: str, thread_ts: str = None) -> str:
    """Get the key for conversation memory lookup."""
    if thread_ts:
        return f"{channel}:{thread_ts}"
    return channel


def get_conversation_history(channel: str, thread_ts: str = None) -> list[dict]:
    """Get conversation history for a thread/DM."""
    key = get_conversation_key(channel, thread_ts)

    # Check if conversation has expired
    if key in MEMORY_TIMESTAMPS:
        if time.time() - MEMORY_TIMESTAMPS[key] > MEMORY_EXPIRY_SECONDS:
            # Conversation expired, clear it
            CONVERSATION_MEMORY.pop(key, None)
            MEMORY_TIMESTAMPS.pop(key, None)
            return []

    return CONVERSATION_MEMORY.get(key, [])


def add_to_conversation(channel: str, thread_ts: str, role: str, content: str):
    """Add a message to conversation history."""
    key = get_conversation_key(channel, thread_ts)

    if key not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[key] = []

    CONVERSATION_MEMORY[key].append({"role": role, "content": content})
    MEMORY_TIMESTAMPS[key] = time.time()

    # Trim to max length
    if len(CONVERSATION_MEMORY[key]) > MAX_HISTORY_LENGTH:
        CONVERSATION_MEMORY[key] = CONVERSATION_MEMORY[key][-MAX_HISTORY_LENGTH:]


def clear_conversation(channel: str, thread_ts: str = None):
    """Clear conversation history for a thread/DM."""
    key = get_conversation_key(channel, thread_ts)
    CONVERSATION_MEMORY.pop(key, None)
    MEMORY_TIMESTAMPS.pop(key, None)


def extract_question(text: str, bot_user_id: str = None) -> str:
    """Extract the question from a message, removing bot mention."""
    # Remove bot mention if present
    if bot_user_id:
        text = re.sub(f'<@{bot_user_id}>', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def format_response_for_slack(response: str) -> str:
    """
    Format the response for Slack markdown.
    Converts standard markdown to Slack's mrkdwn format.
    """
    # Convert **bold** to *bold*
    response = re.sub(r'\*\*(.+?)\*\*', r'*\1*', response)

    # Convert [text](url) to <url|text>
    response = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', response)

    # Convert --- to a line
    response = response.replace('---', '━' * 40)

    return response


def log_question(
    question: str,
    user_id: str,
    source: str,
    channel: str = None,
    response_time_ms: float = None,
    success: bool = True,
    error: str = None
):
    """
    Log a question to the questions log file.

    Args:
        question: The question asked
        user_id: Slack user ID
        source: Where the question came from ('mention', 'dm', 'slash_command')
        channel: Channel ID (optional)
        response_time_ms: How long the response took in milliseconds
        success: Whether the query was successful
        error: Error message if not successful
    """
    try:
        # Load existing log
        log_data = {"questions": []}
        if QUESTIONS_LOG_FILE.exists():
            try:
                with open(QUESTIONS_LOG_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        log_data = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                # File is corrupted or empty, start fresh
                logger.warning("Questions log file was corrupted, starting fresh")

        # Add new entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "user_id": user_id,
            "source": source,
            "channel": channel,
            "response_time_ms": response_time_ms,
            "success": success,
        }
        if error:
            entry["error"] = error

        log_data["questions"].append(entry)

        # Write back
        with open(QUESTIONS_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.debug(f"Logged question from {user_id}: {question[:50]}...")

    except Exception as e:
        logger.error(f"Failed to log question: {e}")


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from text, including Slack-formatted URLs."""
    urls = []

    # Match Slack-formatted URLs: <https://example.com|display text> or <https://example.com>
    slack_url_pattern = r'<(https?://[^|>]+)(?:\|[^>]*)?>|<(https?://[^>]+)>'
    for match in re.finditer(slack_url_pattern, text):
        url = match.group(1) or match.group(2)
        if url:
            urls.append(url)

    # Also match plain URLs
    plain_url_pattern = r'(?<!<)(https?://[^\s<>]+)(?!>)'
    for match in re.finditer(plain_url_pattern, text):
        url = match.group(1)
        if url not in urls:
            urls.append(url)

    return urls


def fetch_webpage_content(url: str) -> dict:
    """
    Fetch webpage content and extract text.
    Returns dict with title, content, and url.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Get title
    title = ''
    if soup.title:
        title = soup.title.string.strip() if soup.title.string else ''

    # Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'iframe']):
        tag.decompose()

    # Try to find main content
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post|entry'))

    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        text = soup.body.get_text(separator='\n', strip=True) if soup.body else soup.get_text(separator='\n', strip=True)

    # Clean up text
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)

    # Get domain for site name
    parsed = urlparse(url)
    site_name = parsed.netloc.replace('www.', '')

    return {
        'url': url,
        'title': title,
        'content': text,
        'site_name': site_name
    }


async def ingest_youtube_video(url: str) -> dict:
    """
    Ingest a YouTube video by downloading transcript and creating embeddings.
    Returns dict with success status and message.
    """
    try:
        # Extract video ID
        video_id = extract_video_id(url)
        if not video_id:
            return {
                'success': False,
                'message': "Could not extract video ID from URL"
            }

        # Get video info
        video_info = await asyncio.to_thread(get_video_info, video_id)
        if not video_info:
            return {
                'success': False,
                'message': "Could not fetch video information"
            }

        # Get transcript
        transcript = await asyncio.to_thread(get_transcript, video_id)
        if not transcript:
            return {
                'success': False,
                'message': f"No transcript available for '{video_info.title}'"
            }

        # Save raw transcript
        await asyncio.to_thread(save_transcript, video_id, transcript)

        # Chunk the transcript
        chunks = await asyncio.to_thread(chunk_transcript, transcript, video_info)
        if not chunks:
            return {
                'success': False,
                'message': "Failed to chunk transcript"
            }

        # Add chunks to embeddings
        manager = get_embeddings_manager()
        num_added = await asyncio.to_thread(manager.add_chunks, chunks)

        return {
            'success': True,
            'message': f"Added '{video_info.title}' ({num_added} chunks)",
            'title': video_info.title,
            'chunks': num_added
        }

    except Exception as e:
        logger.error(f"Error ingesting YouTube video: {e}")
        return {
            'success': False,
            'message': f"Error: {str(e)}"
        }


async def ingest_webpage(url: str) -> dict:
    """
    Ingest a regular webpage by scraping content.
    Returns dict with success status and message.
    """
    try:
        # Fetch the webpage
        page_data = await asyncio.to_thread(fetch_webpage_content, url)

        if len(page_data['content']) < 100:
            return {
                'success': False,
                'message': f"Page has too little content ({len(page_data['content'])} chars)"
            }

        # Create the text for embedding
        text_parts = []
        if page_data['title']:
            text_parts.append(f"# {page_data['title']}")
        text_parts.append(f"\nSource: {page_data['url']}")
        text_parts.append(f"\n{page_data['content']}")

        full_text = '\n'.join(text_parts)

        # Truncate if too long (keep first ~8000 chars to stay within embedding limits)
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "\n\n[Content truncated...]"

        # Create node
        node = TextNode(
            text=full_text,
            metadata={
                'source_type': 'web',
                'url': page_data['url'],
                'page_title': page_data['title'],
                'site_name': page_data['site_name'],
                'content_source': 'slack_ingest',
            },
            excluded_embed_metadata_keys=['url', 'content_source', 'source_type'],
            excluded_llm_metadata_keys=['content_source', 'source_type'],
        )

        # Add to vector store
        manager = get_embeddings_manager()
        manager._index = VectorStoreIndex(
            nodes=[node],
            storage_context=manager.storage_context,
            show_progress=False
        )

        return {
            'success': True,
            'message': f"Added '{page_data['title'] or url}' ({len(page_data['content'])} chars)",
            'title': page_data['title'],
            'content_length': len(page_data['content'])
        }

    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'message': f"Failed to fetch URL: {str(e)}"
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error ingesting URL: {str(e)}"
        }


async def ingest_url(url: str) -> dict:
    """
    Ingest a URL into the knowledge base.
    Automatically detects YouTube URLs and uses full transcript ingestion.
    Returns dict with success status and message.
    """
    # Check if it's a YouTube URL - use full transcript pipeline
    if is_youtube_url(url):
        return await ingest_youtube_video(url)
    else:
        return await ingest_webpage(url)


@app.event("app_mention")
async def handle_mention(event, client, say):
    """Handle @mentions of the bot."""
    global BOT_USER_ID

    channel = event["channel"]
    thread_ts = event.get("thread_ts", event["ts"])
    user_text = event["text"]

    # Get bot user ID if we don't have it
    if BOT_USER_ID is None:
        auth_response = await client.auth_test()
        BOT_USER_ID = auth_response["user_id"]

    # Extract the question
    question = extract_question(user_text, BOT_USER_ID)

    # Check for clear/reset command
    if question.lower() in ['clear', 'reset', 'new conversation', 'start over']:
        clear_conversation(channel, thread_ts)
        await say(
            text="Conversation cleared! Feel free to ask a new question.",
            thread_ts=thread_ts
        )
        return

    if not question:
        await say(
            text="Please ask me a question! For example: @SEOBot What is local SEO?",
            thread_ts=thread_ts
        )
        return

    user_id = event.get("user", "unknown")
    logger.info(f"Mention query from {user_id}: {question[:50]}...")

    # Send "thinking" message
    loading_response = await client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text="Searching the SEO knowledge base..."
    )

    start_time = datetime.now()
    try:
        # Get conversation history for this thread
        history = get_conversation_history(channel, thread_ts)

        # Query the knowledge base with conversation history
        answer = await asyncio.to_thread(query, question, history)

        # Format for Slack
        formatted_answer = format_response_for_slack(answer)

        # Update the loading message with the answer
        await client.chat_update(
            channel=channel,
            ts=loading_response["ts"],
            text=formatted_answer
        )

        # Store in conversation memory (store original question, not with context)
        add_to_conversation(channel, thread_ts, "user", question)
        # Store answer without sources section for cleaner history
        answer_without_sources = answer.split("\n\n---\n**Sources:**")[0]
        add_to_conversation(channel, thread_ts, "assistant", answer_without_sources)

        # Log the question
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_question(
            question=question,
            user_id=user_id,
            source="mention",
            channel=channel,
            response_time_ms=response_time_ms,
            success=True
        )

        logger.info("Mention response sent successfully")

    except Exception as e:
        logger.error(f"Error processing mention: {e}")
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_question(
            question=question,
            user_id=user_id,
            source="mention",
            channel=channel,
            response_time_ms=response_time_ms,
            success=False,
            error=str(e)
        )
        await client.chat_update(
            channel=channel,
            ts=loading_response["ts"],
            text=f"Sorry, I encountered an error: {str(e)}"
        )


@app.event("message")
async def handle_direct_message(event, client, say):
    """Handle direct messages to the bot."""
    global BOT_USER_ID

    # Ignore bot messages to prevent loops
    if event.get("bot_id"):
        return

    text = event.get("text", "").strip()
    if not text:
        return

    # Get bot user ID if we don't have it
    if BOT_USER_ID is None:
        auth_response = await client.auth_test()
        BOT_USER_ID = auth_response["user_id"]

    # Check if this is a DM
    is_dm = event.get("channel_type") == "im"

    # Check for "ingest:" prefix in any channel or DM
    if text.lower().startswith("ingest:") or text.lower().startswith("add:"):
        url_text = text.split(":", 1)[1].strip()
        urls = extract_urls_from_text(url_text)

        if not urls and url_text.startswith('http'):
            urls = [url_text]

        if urls:
            logger.info(f"Message-based ingest: {urls}")

            # React to show we're processing
            try:
                await client.reactions_add(
                    channel=event["channel"],
                    timestamp=event["ts"],
                    name="hourglass_flowing_sand"
                )
            except:
                pass

            results = []
            for url in urls:
                result = await ingest_url(url)
                if result['success']:
                    results.append(f"✅ {result['message']}")
                else:
                    results.append(f"❌ {url}: {result['message']}")

            # Reply in thread
            await client.chat_postMessage(
                channel=event["channel"],
                thread_ts=event["ts"],
                text="*Ingestion Results:*\n" + "\n".join(results)
            )

            # Update reaction
            try:
                await client.reactions_remove(
                    channel=event["channel"],
                    timestamp=event["ts"],
                    name="hourglass_flowing_sand"
                )
                await client.reactions_add(
                    channel=event["channel"],
                    timestamp=event["ts"],
                    name="white_check_mark"
                )
            except:
                pass

            return

    # Only process questions in DMs
    if not is_dm:
        return

    question = text
    user_id = event.get("user", "unknown")
    channel = event["channel"]
    thread_ts = event.get("thread_ts")  # None for non-threaded DMs

    # Check for clear/reset command
    if question.lower() in ['clear', 'reset', 'new conversation', 'start over']:
        clear_conversation(channel, thread_ts)
        await say(
            text="Conversation cleared! Feel free to ask a new question."
        )
        return

    logger.info(f"DM query from {user_id}: {question[:50]}...")

    # Send "thinking" message
    loading_response = await client.chat_postMessage(
        channel=channel,
        text="Searching the SEO knowledge base..."
    )

    start_time = datetime.now()
    try:
        # Get conversation history for this DM
        history = get_conversation_history(channel, thread_ts)

        # Query the knowledge base with conversation history
        answer = await asyncio.to_thread(query, question, history)

        # Format for Slack
        formatted_answer = format_response_for_slack(answer)

        # Update with answer
        await client.chat_update(
            channel=channel,
            ts=loading_response["ts"],
            text=formatted_answer
        )

        # Store in conversation memory
        add_to_conversation(channel, thread_ts, "user", question)
        # Store answer without sources section for cleaner history
        answer_without_sources = answer.split("\n\n---\n**Sources:**")[0]
        add_to_conversation(channel, thread_ts, "assistant", answer_without_sources)

        # Log the question
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_question(
            question=question,
            user_id=user_id,
            source="dm",
            channel=channel,
            response_time_ms=response_time_ms,
            success=True
        )

        logger.info("DM response sent successfully")

    except Exception as e:
        logger.error(f"Error processing DM: {e}")
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_question(
            question=question,
            user_id=user_id,
            source="dm",
            channel=channel,
            response_time_ms=response_time_ms,
            success=False,
            error=str(e)
        )
        await client.chat_update(
            channel=channel,
            ts=loading_response["ts"],
            text=f"Sorry, I encountered an error: {str(e)}"
        )


async def handle_query_with_domain(question: str, domain: str, user_id: str, channel_id: str, respond, command_name: str):
    """Helper function to handle queries with domain support."""
    if not question:
        await respond(
            text=f"Please provide a question! Usage: `/{command_name} What is your question?`"
        )
        return

    logger.info(f"Slash command query ({command_name}) from {user_id}: {question[:50]}...")

    # Send initial response
    domain_name = "all knowledge bases" if domain == "all" or domain is None else DOMAINS.get(domain, {}).get('display_name', domain)
    await respond(text=f"Searching {domain_name}...")

    start_time = datetime.now()
    try:
        # Query the knowledge base with domain
        answer = await asyncio.to_thread(query, question, None, domain)

        # Format for Slack
        formatted_answer = format_response_for_slack(answer)

        # Send the response
        await respond(text=formatted_answer)

        # Log the question
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_question(
            question=question,
            user_id=user_id,
            source=f"slash_command_{command_name}",
            channel=channel_id,
            response_time_ms=response_time_ms,
            success=True
        )

        logger.info(f"Slash command ({command_name}) response sent successfully")

    except Exception as e:
        logger.error(f"Error processing slash command: {e}")
        response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        log_question(
            question=question,
            user_id=user_id,
            source=f"slash_command_{command_name}",
            channel=channel_id,
            response_time_ms=response_time_ms,
            success=False,
            error=str(e)
        )
        await respond(text=f"Sorry, I encountered an error: {str(e)}")


@app.command("/ask")
async def handle_ask_command(ack, command, respond, client):
    """Handle the /ask slash command with auto-routing."""
    await ack()
    question = command.get("text", "").strip()
    await handle_query_with_domain(
        question=question,
        domain=None,  # Auto-route
        user_id=command["user_id"],
        channel_id=command.get("channel_id"),
        respond=respond,
        command_name="ask"
    )


@app.command("/ask-seo")
async def handle_ask_seo_command(ack, command, respond, client):
    """Handle the /ask-seo slash command for SEO-specific queries."""
    await ack()
    question = command.get("text", "").strip()
    await handle_query_with_domain(
        question=question,
        domain="seo",
        user_id=command["user_id"],
        channel_id=command.get("channel_id"),
        respond=respond,
        command_name="ask-seo"
    )


@app.command("/ask-web")
async def handle_ask_web_command(ack, command, respond, client):
    """Handle the /ask-web slash command for AI Website Building queries."""
    await ack()
    question = command.get("text", "").strip()
    await handle_query_with_domain(
        question=question,
        domain="web_builder",
        user_id=command["user_id"],
        channel_id=command.get("channel_id"),
        respond=respond,
        command_name="ask-web"
    )


@app.command("/kb-stats")
async def handle_kb_stats_command(ack, respond):
    """Handle the /kb-stats command to show all knowledge base statistics."""
    await ack()

    try:
        response_lines = ["*Knowledge Base Statistics*\n"]

        for domain_name, domain_config in DOMAINS.items():
            manager = get_embeddings_manager(domain_name)
            stats = manager.get_stats()
            source_stats = manager.get_stats_by_source()

            response_lines.append(f"*{domain_config['display_name']}*")
            response_lines.append(f"  Total Chunks: {stats['total_chunks']:,}")
            if stats['total_chunks'] > 0:
                response_lines.append(f"    - YouTube: {source_stats.get('youtube', 0):,}")
                response_lines.append(f"    - Local Files: {source_stats.get('file', 0):,}")
                response_lines.append(f"    - Local Videos: {source_stats.get('local_video', 0):,}")
                response_lines.append(f"    - Web Pages: {source_stats.get('web', 0):,}")
            response_lines.append("")

        response_lines.append(f"_Embedding Model: text-embedding-3-small_")

        await respond(text="\n".join(response_lines))

    except Exception as e:
        await respond(text=f"Error getting stats: {str(e)}")


@app.command("/seo-stats")
async def handle_stats_command(ack, respond):
    """Handle the /seo-stats command to show SEO knowledge base statistics (legacy)."""
    await ack()

    try:
        manager = get_embeddings_manager("seo")
        stats = manager.get_stats()
        source_stats = manager.get_stats_by_source()

        response = (
            f"*SEO Knowledge Base Statistics*\n\n"
            f"*Total Chunks:* {stats['total_chunks']:,}\n\n"
            f"*Breakdown by Source:*\n"
            f"  YouTube Transcripts: {source_stats.get('youtube', 0):,}\n"
            f"  Local Files: {source_stats.get('file', 0):,}\n"
            f"  Local Videos: {source_stats.get('local_video', 0):,}\n"
            f"  Web Pages: {source_stats.get('web', 0):,}\n\n"
            f"*Embedding Model:* {stats['embedding_model']}\n\n"
            f"_Tip: Use /kb-stats to see all knowledge bases_"
        )

        await respond(text=response)

    except Exception as e:
        await respond(text=f"Error getting stats: {str(e)}")


@app.command("/ingest")
async def handle_ingest_command(ack, command, respond):
    """Handle the /ingest slash command to add URLs to the knowledge base."""
    await ack()

    text = command.get("text", "").strip()
    user_id = command["user_id"]

    if not text:
        await respond(
            text="Please provide a URL to ingest! Usage: `/ingest https://example.com/article`"
        )
        return

    # Extract URLs from the text
    urls = extract_urls_from_text(text)

    if not urls:
        # Try treating the whole text as a URL
        if text.startswith('http'):
            urls = [text]
        else:
            await respond(
                text="No valid URLs found. Please provide a URL starting with http:// or https://"
            )
            return

    logger.info(f"Ingest command from {user_id}: {urls}")

    # Send initial response
    await respond(text=f"Ingesting {len(urls)} URL(s)...")

    results = []
    for url in urls:
        result = await ingest_url(url)
        if result['success']:
            results.append(f"✅ {result['message']}")
        else:
            results.append(f"❌ {url}: {result['message']}")

    # Send final response
    response_text = "*Ingestion Results:*\n" + "\n".join(results)
    await respond(text=response_text)


async def main():
    """Main entry point."""
    # Validate environment variables
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please set these in your .env file or environment.")
        logger.error("See SLACK_SETUP.md for setup instructions.")
        sys.exit(1)

    # Test database connection for all domains
    try:
        for domain_name, domain_config in DOMAINS.items():
            manager = get_embeddings_manager(domain_name)
            stats = manager.get_stats()
            logger.info(f"Connected to {domain_config['display_name']}: {stats['total_chunks']} chunks")
    except Exception as e:
        logger.error(f"Failed to connect to knowledge base: {e}")
        sys.exit(1)

    # Start the bot
    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    logger.info("Starting Multi-Domain Knowledge Base Slack Bot...")
    logger.info("Commands: /ask (auto-route), /ask-seo, /ask-web, /kb-stats")
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
