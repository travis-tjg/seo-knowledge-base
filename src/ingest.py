"""Ingestion pipeline for YouTube transcripts."""

import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Callable

import tiktoken
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from rich.console import Console
import httpx

import sys
sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
from config import TRANSCRIPTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

console = Console()

# Thread-safe lock for database writes
_db_lock = threading.Lock()

# Rate limiting: delay between transcript fetches to avoid YouTube IP blocks
RATE_LIMIT_DELAY = 2.0  # seconds between transcript requests

# Proxy configuration - Oxylabs ISP proxies
PROXIES = [
    {"host": "isp.oxylabs.io", "port": 8001, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8002, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8003, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8004, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8005, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8006, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8007, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8008, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8009, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8010, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8011, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8012, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8013, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8014, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8015, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8016, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8017, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8018, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8019, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
    {"host": "isp.oxylabs.io", "port": 8020, "username": "tjgwebdesign_ERubu", "password": "SF+LBbq2Nk7et"},
]

# Track current proxy index for rotation
_current_proxy_index = 0


def get_next_proxy() -> dict:
    """Get the next proxy in rotation."""
    global _current_proxy_index
    proxy = PROXIES[_current_proxy_index]
    _current_proxy_index = (_current_proxy_index + 1) % len(PROXIES)
    return proxy


def create_proxy_client(proxy: dict) -> httpx.Client:
    """Create an httpx client with the given proxy configuration."""
    proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['host']}:{proxy['port']}"
    return httpx.Client(proxy=proxy_url)


@dataclass
class VideoInfo:
    """Information about a YouTube video."""
    video_id: str
    title: str
    channel_name: str
    video_url: str


@dataclass
class TranscriptChunk:
    """A chunk of transcript with metadata."""
    text: str
    video_id: str
    video_title: str
    channel_name: str
    video_url: str
    start_time: float
    end_time: float
    timestamp_url: str


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Check if it's already a video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    return None


def is_channel_url(url: str) -> bool:
    """Check if URL is a YouTube channel URL."""
    channel_patterns = [
        r'youtube\.com/@[\w-]+',
        r'youtube\.com/c/[\w-]+',
        r'youtube\.com/channel/[\w-]+',
        r'youtube\.com/user/[\w-]+',
    ]
    return any(re.search(pattern, url) for pattern in channel_patterns)


def get_videos_from_channel(channel_url: str) -> list[VideoInfo]:
    """Extract all video information from a YouTube channel."""
    videos = []

    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)

            if info is None:
                console.print(f"[red]Could not extract info from channel: {channel_url}[/red]")
                return videos

            channel_name = info.get('channel', info.get('uploader', 'Unknown Channel'))
            entries = info.get('entries', [])

            for entry in entries:
                if entry is None:
                    continue
                video_id = entry.get('id')
                if video_id:
                    videos.append(VideoInfo(
                        video_id=video_id,
                        title=entry.get('title', 'Unknown Title'),
                        channel_name=channel_name,
                        video_url=f"https://www.youtube.com/watch?v={video_id}"
                    ))

            console.print(f"[green]Found {len(videos)} videos from channel: {channel_name}[/green]")

    except Exception as e:
        console.print(f"[red]Error extracting channel videos: {e}[/red]")

    return videos


def get_video_info(video_id: str) -> Optional[VideoInfo]:
    """Get video information from a video ID."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False
            )

            if info is None:
                return None

            return VideoInfo(
                video_id=video_id,
                title=info.get('title', 'Unknown Title'),
                channel_name=info.get('channel', info.get('uploader', 'Unknown Channel')),
                video_url=f"https://www.youtube.com/watch?v={video_id}"
            )
    except Exception as e:
        console.print(f"[red]Error getting video info for {video_id}: {e}[/red]")
        return None


def get_transcript(video_id: str, use_proxy: bool = True) -> Optional[list[dict]]:
    """Fetch transcript for a video using rotating proxies. Returns None if unavailable."""
    try:
        if use_proxy and PROXIES:
            proxy = get_next_proxy()
            proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['host']}:{proxy['port']}"
            http_client = httpx.Client(proxy=proxy_url)
            api = YouTubeTranscriptApi(http_client=http_client)
            console.print(f"[dim]Using proxy port {proxy['port']}[/dim]")
        else:
            api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id)
        # Convert to list of dicts for compatibility
        return [{'text': item.text, 'start': item.start, 'duration': item.duration} for item in transcript]
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
        console.print(f"[yellow]No transcript available for {video_id}: {type(e).__name__}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error fetching transcript for {video_id}: {e}[/red]")
        return None


def save_transcript(video_id: str, transcript: list[dict]) -> None:
    """Save raw transcript to JSON file."""
    filepath = TRANSCRIPTS_DIR / f"{video_id}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)


def chunk_transcript(
    transcript: list[dict],
    video_info: VideoInfo,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[TranscriptChunk]:
    """
    Chunk transcript into pieces of approximately chunk_size tokens.
    Preserves timestamp information for each chunk.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []

    current_text = ""
    current_tokens = 0
    current_start_time = 0.0
    current_snippets = []  # Track snippets for overlap

    for i, snippet in enumerate(transcript):
        snippet_text = snippet['text'].strip()
        if not snippet_text:
            continue

        snippet_tokens = len(encoding.encode(snippet_text + " "))

        # If adding this snippet exceeds chunk size, save current chunk
        if current_tokens + snippet_tokens > chunk_size and current_text:
            end_time = transcript[i-1]['start'] + transcript[i-1].get('duration', 0)

            chunks.append(TranscriptChunk(
                text=current_text.strip(),
                video_id=video_info.video_id,
                video_title=video_info.title,
                channel_name=video_info.channel_name,
                video_url=video_info.video_url,
                start_time=current_start_time,
                end_time=end_time,
                timestamp_url=f"{video_info.video_url}&t={int(current_start_time)}"
            ))

            # Calculate overlap - keep last snippets up to overlap tokens
            overlap_text = ""
            overlap_tokens = 0
            overlap_snippets = []

            for s in reversed(current_snippets):
                s_tokens = len(encoding.encode(s['text'] + " "))
                if overlap_tokens + s_tokens <= overlap:
                    overlap_snippets.insert(0, s)
                    overlap_text = s['text'] + " " + overlap_text
                    overlap_tokens += s_tokens
                else:
                    break

            # Start new chunk with overlap
            if overlap_snippets:
                current_text = overlap_text
                current_tokens = overlap_tokens
                current_start_time = overlap_snippets[0]['start']
                current_snippets = overlap_snippets.copy()
            else:
                current_text = ""
                current_tokens = 0
                current_start_time = snippet['start']
                current_snippets = []

        # Add current snippet
        if not current_text:
            current_start_time = snippet['start']

        current_text += snippet_text + " "
        current_tokens += snippet_tokens
        current_snippets.append(snippet)

    # Don't forget the last chunk
    if current_text.strip():
        end_time = transcript[-1]['start'] + transcript[-1].get('duration', 0)
        chunks.append(TranscriptChunk(
            text=current_text.strip(),
            video_id=video_info.video_id,
            video_title=video_info.title,
            channel_name=video_info.channel_name,
            video_url=video_info.video_url,
            start_time=current_start_time,
            end_time=end_time,
            timestamp_url=f"{video_info.video_url}&t={int(current_start_time)}"
        ))

    return chunks


def ingest_video(video_id: str) -> list[TranscriptChunk]:
    """Ingest a single video and return chunks."""
    video_info = get_video_info(video_id)
    if not video_info:
        return []

    console.print(f"[blue]Processing: {video_info.title}[/blue]")

    transcript = get_transcript(video_id)
    if not transcript:
        return []

    # Save raw transcript
    save_transcript(video_id, transcript)

    # Chunk the transcript
    chunks = chunk_transcript(transcript, video_info)
    console.print(f"[green]Created {len(chunks)} chunks from {video_info.title}[/green]")

    return chunks


def ingest_url(url: str, rate_limit: bool = True, save_callback=None) -> list[TranscriptChunk]:
    """
    Ingest from a YouTube URL (video or channel).
    Returns all transcript chunks.

    Args:
        url: YouTube video or channel URL
        rate_limit: If True, add delay between transcript fetches to avoid IP blocks
        save_callback: Optional callback function to save chunks after each video.
                      If provided, will be called with chunks after each video is processed.
                      This enables incremental saving to prevent data loss on failure.
    """
    all_chunks = []

    if is_channel_url(url):
        console.print(f"[blue]Ingesting channel: {url}[/blue]")
        videos = get_videos_from_channel(url)

        for i, video in enumerate(videos):
            chunks = ingest_video(video.video_id)
            all_chunks.extend(chunks)

            # Save incrementally if callback provided
            if chunks and save_callback:
                try:
                    save_callback(chunks)
                    console.print(f"[dim]Saved {len(chunks)} chunks to database[/dim]")
                except Exception as e:
                    console.print(f"[red]Error saving chunks: {e}[/red]")

            # Rate limiting between videos to avoid YouTube IP blocks
            if rate_limit and i < len(videos) - 1:
                time.sleep(RATE_LIMIT_DELAY)
    else:
        video_id = extract_video_id(url)
        if video_id:
            chunks = ingest_video(video_id)
            all_chunks.extend(chunks)

            # Save if callback provided
            if chunks and save_callback:
                try:
                    save_callback(chunks)
                    console.print(f"[dim]Saved {len(chunks)} chunks to database[/dim]")
                except Exception as e:
                    console.print(f"[red]Error saving chunks: {e}[/red]")
        else:
            console.print(f"[red]Could not extract video ID from: {url}[/red]")

    console.print(f"[green]Total chunks ingested: {len(all_chunks)}[/green]")
    return all_chunks


def is_youtube_url(text: str) -> bool:
    """Check if text is a YouTube URL."""
    youtube_patterns = [
        r'youtube\.com/watch',
        r'youtu\.be/',
        r'youtube\.com/@',
        r'youtube\.com/c/',
        r'youtube\.com/channel/',
        r'youtube\.com/user/',
    ]
    return any(re.search(pattern, text) for pattern in youtube_patterns)


def process_video_with_proxy(video: VideoInfo, proxy: dict, save_callback: Callable = None) -> tuple[str, int]:
    """
    Process a single video with a specific proxy.
    Returns (video_id, num_chunks) tuple.
    Thread-safe for parallel processing.
    """
    try:
        proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['host']}:{proxy['port']}"
        http_client = httpx.Client(proxy=proxy_url)
        api = YouTubeTranscriptApi(http_client=http_client)

        transcript = api.fetch(video.video_id)
        transcript_data = [{'text': item.text, 'start': item.start, 'duration': item.duration} for item in transcript]

        # Save raw transcript
        save_transcript(video.video_id, transcript_data)

        # Chunk the transcript
        chunks = chunk_transcript(transcript_data, video)

        if chunks and save_callback:
            with _db_lock:
                save_callback(chunks)

        console.print(f"[green]✓ {video.title[:50]}... ({len(chunks)} chunks) [proxy:{proxy['port']}][/green]")
        return (video.video_id, len(chunks))

    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
        console.print(f"[yellow]⚠ {video.title[:50]}... (no transcript)[/yellow]")
        return (video.video_id, 0)
    except Exception as e:
        console.print(f"[red]✗ {video.title[:50]}... ({str(e)[:30]})[/red]")
        return (video.video_id, 0)


def ingest_channel_parallel(url: str, save_callback: Callable = None, max_workers: int = 10) -> int:
    """
    Ingest a YouTube channel using parallel processing with multiple proxies.

    Args:
        url: YouTube channel URL
        save_callback: Callback to save chunks to database
        max_workers: Number of parallel workers (default 10, max 20 for our proxies)

    Returns:
        Total number of chunks ingested
    """
    console.print(f"[blue]Ingesting channel (parallel mode, {max_workers} workers): {url}[/blue]")

    videos = get_videos_from_channel(url)
    if not videos:
        console.print("[red]No videos found[/red]")
        return 0

    total_chunks = 0
    processed = 0

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all videos with rotating proxies
        futures = {}
        for i, video in enumerate(videos):
            proxy = PROXIES[i % len(PROXIES)]
            future = executor.submit(process_video_with_proxy, video, proxy, save_callback)
            futures[future] = video

        # Process completed futures
        for future in as_completed(futures):
            video = futures[future]
            try:
                video_id, num_chunks = future.result()
                total_chunks += num_chunks
                processed += 1

                if processed % 20 == 0:
                    console.print(f"[blue]Progress: {processed}/{len(videos)} videos, {total_chunks} chunks[/blue]")
            except Exception as e:
                console.print(f"[red]Error processing {video.title}: {e}[/red]")
                processed += 1

    console.print(f"[green]Completed: {processed} videos, {total_chunks} total chunks[/green]")
    return total_chunks
