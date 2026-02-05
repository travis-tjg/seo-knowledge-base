#!/usr/bin/env python3
"""
Ingest Digitaleer YouTube channel videos using Whisper transcription.
Downloads audio and transcribes locally since no YouTube captions exist.
"""

import subprocess
import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, '.')

import whisper
import tiktoken
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.embeddings import get_embeddings_manager
from src.ingest import VideoInfo, TranscriptChunk
from config import CHUNK_SIZE, CHUNK_OVERLAP, TRANSCRIPTS_DIR

console = Console()

# Directories
AUDIO_DIR = Path('data/digitaleer_audio')
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Track processed videos
PROCESSED_FILE = Path('data/digitaleer_processed.json')


def load_processed():
    """Load list of already processed video IDs."""
    if PROCESSED_FILE.exists():
        with open(PROCESSED_FILE) as f:
            return set(json.load(f))
    return set()


def save_processed(processed: set):
    """Save list of processed video IDs."""
    with open(PROCESSED_FILE, 'w') as f:
        json.dump(list(processed), f)


def get_channel_videos(url: str) -> list[tuple[str, str]]:
    """Get all video IDs and titles from a channel URL."""
    result = subprocess.run([
        'yt-dlp', '--flat-playlist', '--print', '%(id)s|||%(title)s',
        url
    ], capture_output=True, text=True, timeout=300)

    videos = []
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if '|||' in line:
                vid_id, title = line.split('|||', 1)
                videos.append((vid_id, title))
    return videos


def download_audio(vid_id: str, title: str) -> Path | None:
    """Download audio from a YouTube video."""
    output_path = AUDIO_DIR / f'{vid_id}.mp3'

    if output_path.exists():
        console.print(f"[dim]Audio already exists: {vid_id}[/dim]")
        return output_path

    console.print(f"[blue]Downloading: {title[:50]}...[/blue]")

    result = subprocess.run([
        'yt-dlp',
        '-x', '--audio-format', 'mp3',
        '--audio-quality', '5',
        '-o', str(AUDIO_DIR / f'{vid_id}.%(ext)s'),
        f'https://www.youtube.com/watch?v={vid_id}'
    ], capture_output=True, text=True, timeout=600)

    if result.returncode == 0:
        # Find the actual file
        for f in AUDIO_DIR.glob(f'{vid_id}.*'):
            return f

    console.print(f"[red]Download failed: {result.stderr[:100]}[/red]")
    return None


def transcribe_audio(audio_path: Path, model) -> dict | None:
    """Transcribe audio file using Whisper."""
    try:
        result = model.transcribe(str(audio_path), language='en', verbose=False)
        return result
    except Exception as e:
        console.print(f"[red]Transcription error: {e}[/red]")
        return None


def chunk_whisper_transcript(
    result: dict,
    video_info: VideoInfo,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[TranscriptChunk]:
    """Chunk Whisper transcript into TranscriptChunks."""
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []

    segments = result.get('segments', [])
    if not segments:
        text = result.get('text', '').strip()
        if text:
            chunks.append(TranscriptChunk(
                text=text,
                video_id=video_info.video_id,
                video_title=video_info.title,
                channel_name=video_info.channel_name,
                video_url=video_info.video_url,
                start_time=0.0,
                end_time=0.0,
                timestamp_url=video_info.video_url
            ))
        return chunks

    current_text = ""
    current_tokens = 0
    current_start_time = segments[0]['start']
    current_segments = []

    for segment in segments:
        segment_text = segment['text'].strip()
        if not segment_text:
            continue

        segment_tokens = len(encoding.encode(segment_text + " "))

        if current_tokens + segment_tokens > chunk_size and current_text:
            end_time = current_segments[-1]['end'] if current_segments else segment['start']

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

            # Calculate overlap
            overlap_text = ""
            overlap_tokens = 0
            overlap_segments = []

            for s in reversed(current_segments):
                s_tokens = len(encoding.encode(s['text'] + " "))
                if overlap_tokens + s_tokens <= overlap:
                    overlap_segments.insert(0, s)
                    overlap_text = s['text'] + " " + overlap_text
                    overlap_tokens += s_tokens
                else:
                    break

            if overlap_segments:
                current_text = overlap_text
                current_tokens = overlap_tokens
                current_start_time = overlap_segments[0]['start']
                current_segments = overlap_segments.copy()
            else:
                current_text = ""
                current_tokens = 0
                current_start_time = segment['start']
                current_segments = []

        if not current_text:
            current_start_time = segment['start']

        current_text += segment_text + " "
        current_tokens += segment_tokens
        current_segments.append(segment)

    # Last chunk
    if current_text.strip():
        end_time = current_segments[-1]['end'] if current_segments else 0.0
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


def process_video(vid_id: str, title: str, model, manager) -> int:
    """Process a single video: download, transcribe, chunk, save."""
    # Download
    audio_path = download_audio(vid_id, title)
    if not audio_path:
        return 0

    # Transcribe
    console.print(f"[blue]Transcribing...[/blue]")
    result = transcribe_audio(audio_path, model)
    if not result:
        return 0

    # Save raw transcript
    transcript_path = TRANSCRIPTS_DIR / f"{vid_id}_whisper.json"
    with open(transcript_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Create VideoInfo
    video_info = VideoInfo(
        video_id=vid_id,
        title=title,
        channel_name='Digitaleer',
        video_url=f'https://www.youtube.com/watch?v={vid_id}'
    )

    # Chunk
    chunks = chunk_whisper_transcript(result, video_info)

    if chunks:
        manager.add_chunks(chunks)
        console.print(f"[green]Added {len(chunks)} chunks[/green]")

    # Clean up audio to save space
    try:
        audio_path.unlink()
    except:
        pass

    return len(chunks)


def main():
    """Main ingestion process."""
    console.print("[bold]Digitaleer YouTube Channel Ingestion (Whisper)[/bold]\n")

    # Get manager
    manager = get_embeddings_manager()
    stats = manager.get_stats()
    console.print(f"Starting DB: {stats['total_chunks']} chunks\n")

    # Load Whisper model
    console.print("[blue]Loading Whisper model...[/blue]")
    model = whisper.load_model('base')
    console.print("[green]Model loaded[/green]\n")

    # Get all videos from both tabs
    console.print("[blue]Fetching video list...[/blue]")
    videos = []
    videos.extend(get_channel_videos('https://www.youtube.com/@Digitaleer/videos'))
    videos.extend(get_channel_videos('https://www.youtube.com/@Digitaleer/streams'))

    # Remove duplicates
    seen = set()
    unique_videos = []
    for vid_id, title in videos:
        if vid_id not in seen:
            seen.add(vid_id)
            unique_videos.append((vid_id, title))

    console.print(f"Found {len(unique_videos)} unique videos\n")

    # Load already processed
    processed = load_processed()
    to_process = [(v, t) for v, t in unique_videos if v not in processed]
    console.print(f"Already processed: {len(processed)}")
    console.print(f"To process: {len(to_process)}\n")

    if not to_process:
        console.print("[green]All videos already processed![/green]")
        return

    # Process videos
    total_chunks = 0
    for i, (vid_id, title) in enumerate(to_process):
        console.print(f"\n[bold][{i+1}/{len(to_process)}] {title[:60]}...[/bold]")

        try:
            chunks = process_video(vid_id, title, model, manager)
            total_chunks += chunks

            # Mark as processed
            processed.add(vid_id)
            save_processed(processed)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Progress saved.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    # Final stats
    stats = manager.get_stats()
    console.print(f"\n[bold]Final DB: {stats['total_chunks']} chunks[/bold]")
    console.print(f"[green]Added {total_chunks} chunks this session[/green]")


if __name__ == "__main__":
    main()
