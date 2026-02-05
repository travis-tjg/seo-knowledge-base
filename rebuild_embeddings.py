#!/usr/bin/env python3
"""Rebuild embeddings database from saved transcripts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import LOCAL_DATA_DIR
from src.ingest import chunk_transcript, VideoInfo
from src.embeddings import get_embeddings_manager


def rebuild_from_transcripts(batch_size: int = 100):
    """Rebuild all embeddings from saved transcript JSON files."""
    # Use local transcripts directory
    transcripts_dir = LOCAL_DATA_DIR / "transcripts"
    transcript_files = sorted(transcripts_dir.glob("*.json"))
    print(f"Found {len(transcript_files)} transcript files in {transcripts_dir}")
    sys.stdout.flush()

    manager = get_embeddings_manager()
    total_chunks = 0
    processed = 0
    failed = []
    batch_chunks = []

    for filepath in transcript_files:
        video_id = filepath.stem
        processed += 1

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                transcript = json.load(f)

            if not transcript:
                continue

            # Standard YouTube transcript format
            if isinstance(transcript, list) and len(transcript) > 0:
                video_info = VideoInfo(
                    video_id=video_id,
                    title=f"Video {video_id}",
                    channel_name="Unknown",
                    video_url=f"https://www.youtube.com/watch?v={video_id}"
                )

                chunks = chunk_transcript(transcript, video_info)
                batch_chunks.extend(chunks)

                # Save in batches to reduce API calls
                if len(batch_chunks) >= batch_size:
                    manager.add_chunks(batch_chunks)
                    total_chunks += len(batch_chunks)
                    print(f"Progress: {processed}/{len(transcript_files)} files, {total_chunks} chunks saved")
                    sys.stdout.flush()
                    batch_chunks = []

        except Exception as e:
            failed.append((video_id, str(e)))
            continue

    # Save remaining chunks
    if batch_chunks:
        manager.add_chunks(batch_chunks)
        total_chunks += len(batch_chunks)
        print(f"Final batch: {len(batch_chunks)} chunks saved")

    print(f"\n{'='*60}")
    print(f"Rebuilt {total_chunks} chunks from {processed} transcripts")
    if failed:
        print(f"Failed: {len(failed)} transcripts")
        for vid, err in failed[:10]:
            print(f"  - {vid}: {err}")

    # Get final stats
    stats = manager.get_stats()
    print(f"\nDatabase stats: {stats}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=100, help="Batch size for embedding")
    args = parser.parse_args()
    rebuild_from_transcripts(batch_size=args.batch)
