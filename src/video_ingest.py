"""Video ingestion pipeline using local Whisper for transcription."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tiktoken
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

import sys
sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
from config import CHUNK_SIZE, CHUNK_OVERLAP, TRANSCRIPTS_DIR

console = Console()

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv', '.flv'}

# Whisper model to use - options: tiny, base, small, medium, large
# Larger = more accurate but slower
WHISPER_MODEL = "base"  # Good balance of speed/accuracy for English


@dataclass
class VideoChunk:
    """A chunk of video transcript with metadata."""
    text: str
    file_path: str
    file_name: str
    start_time: float
    end_time: float
    chunk_index: int


def get_whisper_model():
    """Load the Whisper model (cached after first load)."""
    try:
        import whisper
        console.print(f"[blue]Loading Whisper model '{WHISPER_MODEL}'...[/blue]")
        model = whisper.load_model(WHISPER_MODEL)
        console.print(f"[green]Whisper model loaded[/green]")
        return model
    except ImportError:
        console.print("[red]Whisper not installed. Run: pip install openai-whisper[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error loading Whisper model: {e}[/red]")
        return None


# Cache the model
_whisper_model = None


def transcribe_video(file_path: str) -> Optional[dict]:
    """
    Transcribe a video file using Whisper.
    Returns the transcription result with segments.
    """
    global _whisper_model

    if _whisper_model is None:
        _whisper_model = get_whisper_model()
        if _whisper_model is None:
            return None

    try:
        console.print(f"[blue]Transcribing: {Path(file_path).name}[/blue]")
        console.print("[dim]This may take a while for long videos...[/dim]")

        # Transcribe with word timestamps for better chunking
        result = _whisper_model.transcribe(
            file_path,
            language="en",
            verbose=False
        )

        return result
    except Exception as e:
        console.print(f"[red]Error transcribing {file_path}: {e}[/red]")
        return None


def save_transcript(file_path: str, result: dict) -> None:
    """Save the raw transcript to a JSON file."""
    import json

    video_name = Path(file_path).stem
    transcript_path = TRANSCRIPTS_DIR / f"{video_name}_whisper.json"

    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    console.print(f"[dim]Saved transcript to {transcript_path.name}[/dim]")


def chunk_transcript(
    result: dict,
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[VideoChunk]:
    """
    Chunk the Whisper transcript into pieces.
    Uses segment timestamps for accurate time references.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []
    file_name = Path(file_path).name

    segments = result.get('segments', [])
    if not segments:
        # Fallback: create a single chunk from the full text
        text = result.get('text', '').strip()
        if text:
            chunks.append(VideoChunk(
                text=text,
                file_path=file_path,
                file_name=file_name,
                start_time=0.0,
                end_time=0.0,
                chunk_index=0
            ))
        return chunks

    current_text = ""
    current_tokens = 0
    current_start_time = segments[0]['start']
    chunk_index = 0
    current_segments = []

    for segment in segments:
        segment_text = segment['text'].strip()
        if not segment_text:
            continue

        segment_tokens = len(encoding.encode(segment_text + " "))

        # If adding this segment exceeds chunk size, save current chunk
        if current_tokens + segment_tokens > chunk_size and current_text:
            end_time = current_segments[-1]['end'] if current_segments else segment['start']

            chunks.append(VideoChunk(
                text=current_text.strip(),
                file_path=file_path,
                file_name=file_name,
                start_time=current_start_time,
                end_time=end_time,
                chunk_index=chunk_index
            ))
            chunk_index += 1

            # Calculate overlap - keep last segments up to overlap tokens
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

            # Start new chunk with overlap
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

        # Add current segment
        if not current_text:
            current_start_time = segment['start']

        current_text += segment_text + " "
        current_tokens += segment_tokens
        current_segments.append(segment)

    # Don't forget the last chunk
    if current_text.strip():
        end_time = current_segments[-1]['end'] if current_segments else 0.0
        chunks.append(VideoChunk(
            text=current_text.strip(),
            file_path=file_path,
            file_name=file_name,
            start_time=current_start_time,
            end_time=end_time,
            chunk_index=chunk_index
        ))

    return chunks


def ingest_video(file_path: str) -> list[VideoChunk]:
    """Ingest a single video file and return chunks."""
    path = Path(file_path)

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        console.print(f"[yellow]Unsupported video format: {path.suffix}[/yellow]")
        return []

    # Transcribe
    result = transcribe_video(file_path)
    if not result:
        return []

    # Save raw transcript
    save_transcript(file_path, result)

    # Chunk the transcript
    chunks = chunk_transcript(result, file_path)

    if chunks:
        console.print(f"[green]Created {len(chunks)} chunks from {path.name}[/green]")

    return chunks


def ingest_video_folder(
    folder_path: str,
    recursive: bool = True,
    save_callback=None
) -> list[VideoChunk]:
    """
    Ingest all video files from a folder.

    Args:
        folder_path: Path to the folder
        recursive: Whether to search subdirectories
        save_callback: Optional callback to save chunks after each video

    Returns:
        List of all chunks
    """
    path = Path(folder_path)
    if not path.exists():
        console.print(f"[red]Folder not found: {folder_path}[/red]")
        return []

    all_chunks = []

    # Find all video files
    if recursive:
        files = []
        for ext in VIDEO_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
    else:
        files = []
        for ext in VIDEO_EXTENSIONS:
            files.extend(path.glob(f"*{ext}"))

    console.print(f"[blue]Found {len(files)} video files to process[/blue]")

    for i, file_path in enumerate(files):
        try:
            console.print(f"\n[bold]Processing video {i + 1}/{len(files)}[/bold]")
            chunks = ingest_video(str(file_path))
            all_chunks.extend(chunks)

            # Save incrementally if callback provided
            if chunks and save_callback:
                try:
                    save_callback(chunks)
                    console.print(f"[dim]Saved {len(chunks)} chunks to database[/dim]")
                except Exception as e:
                    console.print(f"[red]Error saving chunks: {e}[/red]")

        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {e}[/red]")

    console.print(f"\n[green]Total chunks from videos: {len(all_chunks)}[/green]")
    return all_chunks


def is_video_path(text: str) -> bool:
    """Check if text is a path to a video file or folder containing videos."""
    if not text.startswith('/') and not text.startswith('~'):
        return False

    path = Path(os.path.expanduser(text))

    # Check if it's a video file
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return True

    # Check if it's a folder (could contain videos)
    if path.is_dir():
        # Check if folder name suggests videos
        name_lower = path.name.lower()
        if any(x in name_lower for x in ['video', 'mastermind', 'recording', 'movie']):
            return True

    return False
