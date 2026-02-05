#!/usr/bin/env python3
"""Re-ingest from saved transcripts then continue with remaining channels."""

import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

from config import TRANSCRIPTS_DIR
from src.ingest import (
    TranscriptChunk, chunk_transcript, VideoInfo,
    ingest_channel_parallel, get_videos_from_channel
)
from src.embeddings import get_embeddings_manager

# Channels to process
CHANNELS = [
    "https://www.youtube.com/@SEOFightClub/streams",
    "https://www.youtube.com/@Digitaleer",
    "https://www.youtube.com/@GBPRoast",
]

def reingest_from_transcripts():
    """Re-ingest chunks from saved JSON transcripts."""
    manager = get_embeddings_manager()

    transcript_files = list(TRANSCRIPTS_DIR.glob("*.json"))
    # Filter out whisper transcripts (they have _whisper suffix)
    youtube_transcripts = [f for f in transcript_files if not f.stem.endswith('_whisper')]

    print(f"\nFound {len(youtube_transcripts)} saved YouTube transcripts")

    total_chunks = 0
    batch = []
    batch_size = 50  # Process in batches for efficiency

    for i, transcript_file in enumerate(youtube_transcripts):
        try:
            video_id = transcript_file.stem

            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)

            # Create minimal VideoInfo (we don't have full metadata)
            video_info = VideoInfo(
                video_id=video_id,
                title=f"Video {video_id}",  # We'll get real title from YouTube if needed
                channel_name="Unknown",
                video_url=f"https://www.youtube.com/watch?v={video_id}"
            )

            chunks = chunk_transcript(transcript_data, video_info)
            batch.extend(chunks)

            # Save batch periodically
            if len(batch) >= batch_size * 30:  # ~1500 chunks per batch
                manager.add_chunks(batch)
                total_chunks += len(batch)
                print(f"Progress: {i+1}/{len(youtube_transcripts)} files, {total_chunks} chunks")
                batch = []

        except Exception as e:
            print(f"Error processing {transcript_file}: {e}")

    # Save remaining batch
    if batch:
        manager.add_chunks(batch)
        total_chunks += len(batch)

    print(f"\nRe-ingested {total_chunks} chunks from {len(youtube_transcripts)} transcripts")
    return total_chunks

def ingest_remaining_channels():
    """Ingest from YouTube channels that weren't fully processed."""
    manager = get_embeddings_manager()

    # Get list of already-ingested video IDs from transcripts
    existing_ids = {f.stem for f in TRANSCRIPTS_DIR.glob("*.json") if not f.stem.endswith('_whisper')}
    print(f"\nAlready have {len(existing_ids)} video transcripts")

    def save_chunks(chunks):
        manager.add_chunks(chunks)

    for channel_url in CHANNELS:
        print(f"\n{'='*60}")
        print(f"Processing: {channel_url}")
        print('='*60)

        # Get videos from channel
        videos = get_videos_from_channel(channel_url)

        # Filter out already-processed videos
        new_videos = [v for v in videos if v.video_id not in existing_ids]
        print(f"Found {len(videos)} total videos, {len(new_videos)} new to process")

        if new_videos:
            # Process only new videos in parallel
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from src.ingest import process_video_with_proxy, PROXIES, _db_lock

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                for i, video in enumerate(new_videos):
                    proxy = PROXIES[i % len(PROXIES)]
                    future = executor.submit(process_video_with_proxy, video, proxy, save_chunks)
                    futures[future] = video

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error: {e}")

def main():
    print("="*60)
    print("SEO Knowledge Base - Re-ingest and Continue")
    print("="*60)

    # Step 1: Re-ingest from saved transcripts (fast)
    print("\n[Step 1] Re-ingesting from saved transcripts...")
    reingest_from_transcripts()

    # Step 2: Continue with remaining videos from channels
    print("\n[Step 2] Fetching remaining videos from channels...")
    ingest_remaining_channels()

    # Final stats
    manager = get_embeddings_manager()
    stats = manager.get_stats_by_source()
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"Final DB: YouTube={stats.get('youtube', 0)}, Videos={stats.get('local_video', 0)}, Total={stats['total']}")

if __name__ == "__main__":
    main()
