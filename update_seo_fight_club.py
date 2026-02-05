#!/usr/bin/env python3
"""
Weekly update script for SEO Fight Club videos.

Checks for new videos from the SEO Fight Club channel and ingests any
that aren't already in the database.

Run manually or schedule for Tuesday nights after the show.
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

# Proxy rotation
_current_proxy_index = 0


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


def get_transcript(video_id: str, use_proxy: bool = True) -> list[dict] | None:
    """Fetch transcript for a video using proxy. Returns None if unavailable."""
    try:
        if use_proxy and PROXIES:
            proxy = get_next_proxy()
            proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['host']}:{proxy['port']}"
            http_client = httpx.Client(proxy=proxy_url)
            api = YouTubeTranscriptApi(http_client=http_client)
        else:
            api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id)
        return [{'text': item.text, 'start': item.start, 'duration': item.duration} for item in transcript]
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
        console.print(f"[yellow]No transcript available for {video_id}: {type(e).__name__}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error fetching transcript for {video_id}: {e}[/red]")
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


def main(latest_only: bool = False, max_videos: int = None):
    """
    Main update process.

    Args:
        latest_only: If True, only process the most recent new video
        max_videos: Maximum number of new videos to process (None = all)
    """
    console.print(f"\n[bold]SEO Fight Club Weekly Update[/bold]")
    console.print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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

    console.print(f"Found {len(videos)} total videos\n")

    # Load already processed
    processed = load_processed()
    console.print(f"Already processed: {len(processed)} videos")

    # Find new videos (first in list is most recent)
    new_videos = [v for v in videos if v.video_id not in processed]
    console.print(f"New videos to process: {len(new_videos)}\n")

    if not new_videos:
        console.print("[green]No new videos to ingest![/green]")
        return

    # Limit videos if requested
    if latest_only:
        new_videos = new_videos[:1]
        console.print("[blue]Processing only the latest video[/blue]\n")
    elif max_videos:
        new_videos = new_videos[:max_videos]
        console.print(f"[blue]Processing up to {max_videos} videos[/blue]\n")

    # Process new videos
    total_chunks = 0
    for i, video in enumerate(new_videos):
        console.print(f"\n[{i+1}/{len(new_videos)}] {video.title[:60]}...")

        try:
            chunks = ingest_video(video, manager)
            total_chunks += chunks

            # Mark as processed
            processed.add(video.video_id)
            save_processed(processed)

            # Rate limit between videos
            if i < len(new_videos) - 1:
                time.sleep(2)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Progress saved.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    # Final stats
    stats = manager.get_stats()
    console.print(f"\n[bold]Update Complete![/bold]")
    console.print(f"Added {total_chunks} new chunks")
    console.print(f"Database now has {stats['total_chunks']:,} total chunks")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update SEO Fight Club videos in knowledge base")
    parser.add_argument("--latest", action="store_true", help="Only process the latest new video")
    parser.add_argument("--max", type=int, help="Maximum number of videos to process")
    parser.add_argument("--all", action="store_true", help="Process all new videos (default)")
    args = parser.parse_args()

    main(latest_only=args.latest, max_videos=args.max)
