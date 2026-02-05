#!/usr/bin/env python3
"""Script to ingest remaining TestedSEO .mov files."""

import sys
sys.path.insert(0, '.')

from src.video_ingest import ingest_video
from src.embeddings import get_embeddings_manager
from rich.console import Console

console = Console()

# Files that still need to be transcribed
VIDEOS_TO_INGEST = [
    "/Users/travisgodec/Library/CloudStorage/GoogleDrive-travisgodec@gmail.com/My Drive/Macbook Pro/TestedSEO/Week 6/Screen Recording 2025-10-29 at 11.04.29 AM.mov",
    "/Users/travisgodec/Library/CloudStorage/GoogleDrive-travisgodec@gmail.com/My Drive/Macbook Pro/TestedSEO/Week 8/Screen Recording 2025-11-12 at 11.02.31 AM.mov",
    "/Users/travisgodec/Library/CloudStorage/GoogleDrive-travisgodec@gmail.com/My Drive/Macbook Pro/TestedSEO/Week 9/Screen Recording 2025-11-19 at 11.02.45 AM.mov",
]

def main():
    manager = get_embeddings_manager()

    # Show starting stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Starting DB stats:[/bold]")
    console.print(f"  YouTube: {stats.get('youtube', 0)}")
    console.print(f"  Local Videos: {stats.get('local_video', 0)}")
    console.print(f"  Files: {stats.get('file', 0)}")
    console.print(f"  Total: {stats['total']}")

    total_chunks = 0

    for i, video_path in enumerate(VIDEOS_TO_INGEST, 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Processing video {i}/{len(VIDEOS_TO_INGEST)}[/bold]")
        console.print(f"[blue]{video_path.split('/')[-1]}[/blue]")
        console.print('='*60)

        try:
            chunks = ingest_video(video_path)
            if chunks:
                manager.add_chunks(chunks)
                total_chunks += len(chunks)
                console.print(f"[green]Added {len(chunks)} chunks to database[/green]")
        except Exception as e:
            console.print(f"[red]Error processing video: {e}[/red]")

    # Show final stats
    stats = manager.get_stats_by_source()
    console.print(f"\n{'='*60}")
    console.print(f"[bold green]COMPLETE![/bold green]")
    console.print(f"\n[bold]Final DB stats:[/bold]")
    console.print(f"  YouTube: {stats.get('youtube', 0)}")
    console.print(f"  Local Videos: {stats.get('local_video', 0)}")
    console.print(f"  Files: {stats.get('file', 0)}")
    console.print(f"  Total: {stats['total']}")
    console.print(f"\n[bold]New chunks added: {total_chunks}[/bold]")

if __name__ == "__main__":
    main()
