#!/usr/bin/env python3
"""Script to ingest Desktop .mov files by converting to MP3 first."""

import os
import subprocess
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, '.')

from src.video_ingest import transcribe_video, chunk_transcript, save_transcript
from src.embeddings import get_embeddings_manager
from rich.console import Console

console = Console()

# Desktop folder
DESKTOP_FOLDER = "/Users/travisgodec/Desktop"


def find_video_files(folder: str) -> list[str]:
    """Find all video files in folder (not recursive for Desktop)."""
    video_extensions = {'.mov', '.mp4', '.mkv', '.webm', '.m4v', '.avi'}
    video_files = []
    for f in os.listdir(folder):
        if Path(f).suffix.lower() in video_extensions:
            video_files.append(os.path.join(folder, f))
    return video_files


def get_files_already_in_db(manager) -> set[str]:
    """Get set of filenames already in the database."""
    results = manager.collection.get(include=['metadatas'])
    files_in_db = set()
    for meta in results['metadatas']:
        if meta:
            fn = meta.get('file_name', '')
            files_in_db.add(fn)
    return files_in_db


def convert_to_mp3(video_path: str, output_path: str) -> bool:
    """Convert video to MP3 using ffmpeg."""
    try:
        console.print(f"[blue]Converting to MP3...[/blue]")
        result = subprocess.run([
            'ffmpeg', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-ab', '128k',  # 128kbps is plenty for speech
            '-ar', '16000',  # 16kHz sample rate (Whisper's native rate)
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            output_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            # Show size comparison
            orig_size = os.path.getsize(video_path) / (1024 * 1024 * 1024)
            new_size = os.path.getsize(output_path) / (1024 * 1024)
            console.print(f"[green]Converted: {orig_size:.1f} GB -> {new_size:.1f} MB[/green]")
            return True
        else:
            console.print(f"[red]FFmpeg error: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Conversion error: {e}[/red]")
        return False


def main():
    manager = get_embeddings_manager()

    # Show starting stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Starting DB stats:[/bold]")
    console.print(f"  YouTube: {stats.get('youtube', 0)}")
    console.print(f"  Local Videos: {stats.get('local_video', 0)}")
    console.print(f"  Files: {stats.get('file', 0)}")
    console.print(f"  Total: {stats['total']}")

    # Find all video files on Desktop
    all_video_files = find_video_files(DESKTOP_FOLDER)
    console.print(f"\n[blue]Found {len(all_video_files)} video files on Desktop[/blue]")

    # Get files already in DB
    files_in_db = get_files_already_in_db(manager)
    console.print(f"[blue]Files already in database: {len(files_in_db)}[/blue]")

    # Filter to only files not yet processed
    videos_to_process = []
    for path in all_video_files:
        filename = os.path.basename(path)
        if filename not in files_in_db:
            videos_to_process.append(path)
        else:
            console.print(f"[dim]Skipping (already in DB): {filename}[/dim]")

    console.print(f"\n[bold]Videos to process: {len(videos_to_process)}[/bold]")
    for v in videos_to_process:
        size_gb = os.path.getsize(v) / (1024 * 1024 * 1024)
        console.print(f"  - {os.path.basename(v)} ({size_gb:.1f} GB)")

    total_chunks = 0

    for i, video_path in enumerate(videos_to_process, 1):
        video_name = Path(video_path).stem
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Processing video {i}/{len(videos_to_process)}[/bold]")
        console.print(f"[blue]{video_name}[/blue]")
        console.print('='*60)

        try:
            # Create temp MP3 file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                mp3_path = tmp.name

            # Convert to MP3
            if not convert_to_mp3(video_path, mp3_path):
                console.print(f"[red]Skipping due to conversion error[/red]")
                if os.path.exists(mp3_path):
                    os.unlink(mp3_path)
                continue

            # Transcribe the MP3
            console.print(f"[blue]Transcribing audio...[/blue]")
            result = transcribe_video(mp3_path)

            if not result:
                console.print(f"[red]Transcription failed[/red]")
                os.unlink(mp3_path)
                continue

            # Save transcript (use original video path for naming)
            save_transcript(video_path, result)

            # Chunk the transcript (use original video path for metadata)
            chunks = chunk_transcript(result, video_path)

            if chunks:
                manager.add_chunks(chunks)
                total_chunks += len(chunks)
                console.print(f"[green]Added {len(chunks)} chunks to database[/green]")

            # Clean up temp file
            os.unlink(mp3_path)

        except Exception as e:
            console.print(f"[red]Error processing video: {e}[/red]")
            import traceback
            traceback.print_exc()

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
