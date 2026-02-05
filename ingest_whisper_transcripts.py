#!/usr/bin/env python3
"""Ingest Whisper transcripts (TestedSEO, Digitaleer audio, etc.) into the database."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from config import LOCAL_DATA_DIR
from src.video_ingest import VideoChunk
from src.embeddings import get_embeddings_manager
from src.semantic_chunker import chunk_transcript
import tiktoken


def chunk_text_fixed(text: str, chunk_size: int = 500, overlap: int = 100) -> list[tuple[str, int]]:
    """Chunk text into pieces of approximately chunk_size tokens. Returns list of (text, start_idx)."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append((chunk_text, start))

        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100, use_semantic: bool = True) -> list[tuple[str, int]]:
    """
    Chunk text into pieces. Returns list of (text, start_idx).

    Args:
        use_semantic: If True, uses semantic chunking (respects sentence/paragraph boundaries).
                      If False, uses fixed-size token chunking.
    """
    if use_semantic:
        return chunk_transcript(text, max_tokens=chunk_size, min_tokens=100)
    else:
        return chunk_text_fixed(text, chunk_size=chunk_size, overlap=overlap)


def ingest_whisper_transcripts(batch_size: int = 50, boost: float = 1.0, use_semantic: bool = True):
    """Ingest all Whisper transcripts into the database.

    Args:
        batch_size: Number of chunks to embed in each batch
        boost: Relevance boost multiplier (e.g., 1.15 for 15% boost)
        use_semantic: If True, use semantic chunking; otherwise use fixed-size
    """
    transcripts_dir = LOCAL_DATA_DIR / "transcripts"
    whisper_files = sorted([f for f in transcripts_dir.glob("*_whisper.json")])

    print(f"Found {len(whisper_files)} Whisper transcript files")
    sys.stdout.flush()

    manager = get_embeddings_manager()
    total_chunks = 0
    processed = 0
    failed = []
    batch_chunks = []

    for filepath in whisper_files:
        video_id = filepath.stem.replace("_whisper", "")
        processed += 1

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Whisper format has a "text" field with full transcript
            if isinstance(data, dict) and 'text' in data:
                full_text = data['text']
            elif isinstance(data, str):
                full_text = data
            else:
                print(f"Unknown format for {filepath.name}")
                continue

            if not full_text or len(full_text) < 100:
                continue

            # Chunk the transcript
            text_chunks = chunk_text(full_text, use_semantic=use_semantic)

            for chunk_text_content, start_idx in text_chunks:
                chunk = VideoChunk(
                    text=chunk_text_content,
                    file_path=str(filepath),
                    file_name=video_id,
                    start_time=float(start_idx),  # Use token position as proxy for time
                    end_time=float(start_idx + 500),
                    chunk_index=len(batch_chunks)
                )
                batch_chunks.append(chunk)

            # Save in batches
            if len(batch_chunks) >= batch_size:
                manager.add_chunks(batch_chunks, boost=boost)
                total_chunks += len(batch_chunks)
                print(f"Progress: {processed}/{len(whisper_files)} files, {total_chunks} chunks saved")
                sys.stdout.flush()
                batch_chunks = []

        except Exception as e:
            failed.append((video_id, str(e)))
            continue

    # Save remaining chunks
    if batch_chunks:
        manager.add_chunks(batch_chunks, boost=boost)
        total_chunks += len(batch_chunks)
        print(f"Final batch: {len(batch_chunks)} chunks saved")

    print(f"\n{'='*60}")
    print(f"Ingested {total_chunks} chunks from {processed} Whisper transcripts")
    if failed:
        print(f"Failed: {len(failed)} transcripts")
        for vid, err in failed[:10]:
            print(f"  - {vid}: {err}")

    # Get final stats
    stats = manager.get_stats_by_source()
    print(f"\nDatabase breakdown:")
    for source, count in stats.items():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=50, help="Batch size for embedding")
    parser.add_argument("--boost", type=float, default=1.0,
                        help="Relevance boost multiplier (e.g., 1.15 for 15%% boost)")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Use fixed-size chunking instead of semantic chunking")
    args = parser.parse_args()
    ingest_whisper_transcripts(
        batch_size=args.batch,
        boost=args.boost,
        use_semantic=not args.no_semantic
    )
