#!/usr/bin/env python3
"""
Retry failed SEO Fight Club videos.

Removes specified video IDs from the processed list and re-attempts ingestion.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import httpx
from rich.console import Console
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

from src.ingest import get_videos_from_channel, VideoInfo, chunk_transcript, save_transcript, PROXIES
from src.embeddings import get_embeddings_manager

console = Console()

# SEO Fight Club channel URL
SEO_FIGHT_CLUB_CHANNEL = "https://www.youtube.com/@SEOFightClub/streams"

# Track processed videos
PROCESSED_FILE = Path(__file__).parent / "data" / "seo_fight_club_processed.json"

# Videos that failed - these will be removed from processed and retried
FAILED_VIDEO_IDS = [
    "3RDkkYANBB8", "3pbSbnp-xFs", "4feG_owKEyw", "5sTw7ye46sI", "83O6Mjf5-pc",
    "9_wXjVtQMBY", "A3YblPAp4pw", "AiohB_TvUPE", "GF7yKUa9pi4", "HBfQxYieKAg",
    "Hc1jT8EHK6M", "JTbSfjA0tQA", "JUwolh4_Xns", "LTEE85nPLtU", "NX7ho9MrCR4",
    "ONqPzXWE1F4", "R8QXMvvSCJA", "RWWEqwdC_cI", "ZRZ0zt2niuw", "ZwxabfInDl0",
    "berhWEuck9c", "dGPJdXoNcFQ", "iI1VQw2qZII", "j7bXmv3jIE0", "jLKM8kUtihw",
    "o0VKBzE1p_E", "uCVsuVkf8es", "uVLiCu1GiUY", "uv941AZEi4o", "uwNz_-ezWJA",
    "w1zLmo_Mvmc", "yEc8cWwN_QQ", "yNb0MVEhkaE", "ybxu6-nIiuU",
]

# Proxy rotation - start from a different index for variety
_current_proxy_index = 10


def get_next_proxy() -> dict:
    """Get the next proxy in rotation."""
    global _current_proxy_index
    proxy = PROXIES[_current_proxy_index]
    _current_proxy_index = (_current_proxy_index + 1) % len(PROXIES)
    return proxy


def load_processed() -> set:
    """Load list of already processed video IDs."""
    if PROCESSED_FILE.exists():
        try:
            with open(PROCESSED_FILE) as f:
                content = f.read().strip()
                if content:
                    return set(json.loads(content))
        except (json.JSONDecodeError, ValueError):
            console.print("[yellow]Processed file was corrupted, starting fresh[/yellow]")
    return set()


def save_processed(processed: set):
    """Save list of processed video IDs."""
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_FILE, 'w') as f:
        json.dump(list(processed), f, indent=2)


def get_transcript(video_id: str) -> list[dict] | None:
    """Fetch transcript for a video using proxy. Returns None if unavailable."""
    # Try multiple proxies before giving up
    attempts = 3
    for attempt in range(attempts):
        try:
            proxy = get_next_proxy()
            proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['host']}:{proxy['port']}"
            console.print(f"[dim]Trying proxy port {proxy['port']} (attempt {attempt + 1}/{attempts})...[/dim]")
            http_client = httpx.Client(proxy=proxy_url, timeout=30.0)
            api = YouTubeTranscriptApi(http_client=http_client)

            transcript = api.fetch(video_id)
            return [{'text': item.text, 'start': item.start, 'duration': item.duration} for item in transcript]
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            console.print(f"[yellow]No transcript available for {video_id}: {type(e).__name__}[/yellow]")
            return None  # Don't retry for videos without transcripts
        except VideoUnavailable as e:
            console.print(f"[yellow]Video unavailable for {video_id}: {type(e).__name__}[/yellow]")
            return None  # Don't retry for unavailable videos
        except Exception as e:
            if attempt < attempts - 1:
                console.print(f"[yellow]Attempt {attempt + 1} failed: {type(e).__name__}, retrying...[/yellow]")
                time.sleep(3)  # Wait before retrying
            else:
                console.print(f"[red]All attempts failed for {video_id}: {e}[/red]")
                return None
    return None


def ingest_video(video: VideoInfo, manager) -> int:
    """Ingest a single video. Returns number of chunks added."""
    console.print(f"[blue]Processing: {video.title}[/blue]")

    transcript = get_transcript(video.video_id)
    if not transcript:
        return 0

    # Save raw transcript
    save_transcript(video.video_id, transcript)

    # Chunk the transcript
    chunks = chunk_transcript(transcript, video)

    if chunks:
        manager.add_chunks(chunks)
        console.print(f"[green]Added {len(chunks)} chunks from: {video.title}[/green]")

    return len(chunks)


def main():
    """Main retry process."""
    console.print(f"\n[bold]SEO Fight Club - Retry Failed Videos[/bold]")
    console.print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load processed and remove failed videos
    processed = load_processed()
    original_count = len(processed)

    removed_count = 0
    for video_id in FAILED_VIDEO_IDS:
        if video_id in processed:
            processed.remove(video_id)
            removed_count += 1

    console.print(f"Removed {removed_count} failed video IDs from processed list")
    save_processed(processed)

    # Get embeddings manager
    manager = get_embeddings_manager()
    stats = manager.get_stats()
    console.print(f"Current database: {stats['total_chunks']:,} chunks\n")

    # Get all videos from channel
    console.print("[blue]Fetching video list from SEO Fight Club...[/blue]")
    videos = get_videos_from_channel(SEO_FIGHT_CLUB_CHANNEL)

    if not videos:
        console.print("[red]No videos found![/red]")
        return

    # Filter to only the failed videos we want to retry
    videos_to_retry = [v for v in videos if v.video_id in FAILED_VIDEO_IDS]
    console.print(f"Found {len(videos_to_retry)} videos to retry\n")

    if not videos_to_retry:
        console.print("[yellow]No matching videos found to retry[/yellow]")
        return

    # Process retry videos
    total_chunks = 0
    successful = 0
    for i, video in enumerate(videos_to_retry):
        console.print(f"\n[{i+1}/{len(videos_to_retry)}] {video.title[:60]}...")

        try:
            chunks = ingest_video(video, manager)
            total_chunks += chunks

            # Mark as processed
            processed.add(video.video_id)
            save_processed(processed)

            if chunks > 0:
                successful += 1

            # Rate limit between videos
            if i < len(videos_to_retry) - 1:
                time.sleep(3)  # Longer delay for retries

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Progress saved.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    # Final stats
    stats = manager.get_stats()
    console.print(f"\n[bold]Retry Complete![/bold]")
    console.print(f"Successfully ingested: {successful}/{len(videos_to_retry)} videos")
    console.print(f"Added {total_chunks} new chunks")
    console.print(f"Database now has {stats['total_chunks']:,} total chunks")


if __name__ == "__main__":
    main()
