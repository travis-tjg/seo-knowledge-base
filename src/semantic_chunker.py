"""Semantic chunking - splits text on natural boundaries instead of arbitrary token counts."""

import re
import tiktoken
from dataclasses import dataclass


@dataclass
class SemanticChunk:
    """A semantically coherent chunk of text."""
    text: str
    start_idx: int  # Character position in original text
    token_count: int


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle common abbreviations to avoid false splits
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.\s', r'\1<DOT> ', text)

    # Split on sentence-ending punctuation followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore dots
    sentences = [s.replace('<DOT>', '.') for s in sentences]

    return [s.strip() for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double newline or more)."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def semantic_chunk(
    text: str,
    max_tokens: int = 500,
    min_tokens: int = 100,
    overlap_sentences: int = 1
) -> list[SemanticChunk]:
    """
    Split text into semantically coherent chunks.

    Unlike fixed-size chunking, this:
    1. Respects paragraph boundaries when possible
    2. Keeps sentences together (never splits mid-sentence)
    3. Aims for target size but prioritizes semantic coherence

    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk (soft limit - may exceed for single sentences)
        min_tokens: Minimum tokens per chunk (will merge small chunks)
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of SemanticChunk objects
    """
    encoding = tiktoken.get_encoding("cl100k_base")

    # First split into paragraphs
    paragraphs = split_into_paragraphs(text)

    chunks = []
    current_chunk = []
    current_tokens = 0
    current_start = 0

    for para in paragraphs:
        para_tokens = len(encoding.encode(para))

        # If paragraph is too long, split into sentences
        if para_tokens > max_tokens:
            sentences = split_into_sentences(para)

            for sentence in sentences:
                sent_tokens = len(encoding.encode(sentence))

                # If adding this sentence exceeds max, save current chunk
                if current_tokens + sent_tokens > max_tokens and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(SemanticChunk(
                        text=chunk_text,
                        start_idx=current_start,
                        token_count=current_tokens
                    ))

                    # Keep last N sentences for overlap
                    if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                        overlap = current_chunk[-overlap_sentences:]
                        current_chunk = overlap
                        current_tokens = len(encoding.encode(' '.join(overlap)))
                    else:
                        current_chunk = []
                        current_tokens = 0
                    current_start = text.find(current_chunk[0]) if current_chunk else text.find(sentence)

                current_chunk.append(sentence)
                current_tokens += sent_tokens
        else:
            # Paragraph fits, check if adding it exceeds max
            if current_tokens + para_tokens > max_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(SemanticChunk(
                    text=chunk_text,
                    start_idx=current_start,
                    token_count=current_tokens
                ))

                # Keep last sentence(s) for overlap
                sentences = split_into_sentences(' '.join(current_chunk))
                if overlap_sentences > 0 and len(sentences) >= overlap_sentences:
                    overlap = sentences[-overlap_sentences:]
                    current_chunk = overlap
                    current_tokens = len(encoding.encode(' '.join(overlap)))
                else:
                    current_chunk = []
                    current_tokens = 0
                current_start = text.find(current_chunk[0]) if current_chunk else text.find(para)

            current_chunk.append(para)
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if current_tokens >= min_tokens or not chunks:
            chunks.append(SemanticChunk(
                text=chunk_text,
                start_idx=current_start,
                token_count=current_tokens
            ))
        elif chunks:
            # Merge with previous chunk if too small
            prev = chunks[-1]
            chunks[-1] = SemanticChunk(
                text=prev.text + ' ' + chunk_text,
                start_idx=prev.start_idx,
                token_count=prev.token_count + current_tokens
            )

    return chunks


def chunk_transcript(
    text: str,
    max_tokens: int = 500,
    min_tokens: int = 100
) -> list[tuple[str, int]]:
    """
    Convenience function for chunking transcripts.
    Returns list of (chunk_text, start_token_position) tuples,
    compatible with existing ingest code.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = semantic_chunk(text, max_tokens=max_tokens, min_tokens=min_tokens)

    result = []
    running_tokens = 0

    for chunk in chunks:
        result.append((chunk.text, running_tokens))
        running_tokens += chunk.token_count

    return result
