#!/usr/bin/env python3
"""Script to ingest all YouTube channels in parallel."""

import sys
sys.path.insert(0, '.')

from src.ingest import ingest_channel_parallel
from src.embeddings import get_embeddings_manager

CHANNELS = [
    "https://www.youtube.com/@SEOFightClub/streams",
    "https://www.youtube.com/@Digitaleer",
    "https://www.youtube.com/@GBPRoast",
]

def main():
    manager = get_embeddings_manager()

    # Show starting stats
    stats = manager.get_stats_by_source()
    print(f"\nStarting DB: YouTube={stats.get('youtube', 0)}, Videos={stats.get('local_video', 0)}, Total={stats['total']}")

    def save_chunks(chunks):
        manager.add_chunks(chunks)

    total = 0
    for channel in CHANNELS:
        print(f"\n{'='*60}")
        print(f"Processing: {channel}")
        print('='*60)
        chunks = ingest_channel_parallel(channel, save_callback=save_chunks, max_workers=10)
        total += chunks

    # Show final stats
    stats = manager.get_stats_by_source()
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"Final DB: YouTube={stats.get('youtube', 0)}, Videos={stats.get('local_video', 0)}, Total={stats['total']}")
    print(f"New chunks added: {total}")

if __name__ == "__main__":
    main()
