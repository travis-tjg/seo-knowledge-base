"""File ingestion pipeline for local documents (PDF, TXT, MD, DOCX, HTML)."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tiktoken
from rich.console import Console

import sys
sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
from config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.html', '.htm'}


@dataclass
class FileChunk:
    """A chunk of file content with metadata."""
    text: str
    file_path: str
    file_name: str
    page_number: Optional[int]  # For PDFs
    chunk_index: int


def extract_text_from_pdf(file_path: str) -> list[tuple[int, str]]:
    """Extract text from PDF, returns list of (page_number, text) tuples."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(file_path)
        pages = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages.append((page_num, text))
        doc.close()
        return pages
    except Exception as e:
        console.print(f"[red]Error reading PDF {file_path}: {e}[/red]")
        return []


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n\n'.join(paragraphs)
    except Exception as e:
        console.print(f"[red]Error reading DOCX {file_path}: {e}[/red]")
        return ""


def extract_text_from_html(file_path: str) -> str:
    """Extract text from HTML file."""
    try:
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n')
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        console.print(f"[red]Error reading HTML {file_path}: {e}[/red]")
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from plain text or markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        console.print(f"[red]Error reading text file {file_path}: {e}[/red]")
        return ""


def chunk_text(
    text: str,
    file_path: str,
    page_number: Optional[int] = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_tokens: int = 7000  # Stay well under 8192 embedding limit
) -> list[FileChunk]:
    """Chunk text into pieces of approximately chunk_size tokens."""
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []
    file_name = Path(file_path).name

    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)

    current_text = ""
    current_tokens = 0
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = len(encoding.encode(para + "\n\n"))

        # If paragraph itself is too large, split it by sentences
        if para_tokens > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_tokens = len(encoding.encode(sent + " "))

                # If single sentence is still too large, split by words
                if sent_tokens > max_tokens:
                    words = sent.split()
                    word_chunk = ""
                    word_tokens = 0
                    for word in words:
                        wt = len(encoding.encode(word + " "))
                        if word_tokens + wt > max_tokens and word_chunk:
                            if current_text:
                                chunks.append(FileChunk(
                                    text=current_text.strip(),
                                    file_path=file_path,
                                    file_name=file_name,
                                    page_number=page_number,
                                    chunk_index=chunk_index
                                ))
                                chunk_index += 1
                                current_text = ""
                                current_tokens = 0
                            current_text = word_chunk + " "
                            current_tokens = word_tokens
                            word_chunk = word
                            word_tokens = wt
                        else:
                            word_chunk += " " + word if word_chunk else word
                            word_tokens += wt
                    if word_chunk:
                        current_text += word_chunk + " "
                        current_tokens += word_tokens
                    continue

                if current_tokens + sent_tokens > chunk_size and current_text:
                    chunks.append(FileChunk(
                        text=current_text.strip(),
                        file_path=file_path,
                        file_name=file_name,
                        page_number=page_number,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                    current_text = ""
                    current_tokens = 0

                current_text += sent + " "
                current_tokens += sent_tokens
            continue

        # If adding this paragraph exceeds chunk size, save current chunk
        if current_tokens + para_tokens > chunk_size and current_text:
            chunks.append(FileChunk(
                text=current_text.strip(),
                file_path=file_path,
                file_name=file_name,
                page_number=page_number,
                chunk_index=chunk_index
            ))
            chunk_index += 1

            # Keep overlap - take last portion of current text
            overlap_text = ""
            overlap_tokens = 0
            words = current_text.split()
            for word in reversed(words):
                word_tokens = len(encoding.encode(word + " "))
                if overlap_tokens + word_tokens <= overlap:
                    overlap_text = word + " " + overlap_text
                    overlap_tokens += word_tokens
                else:
                    break

            current_text = overlap_text
            current_tokens = overlap_tokens

        current_text += para + "\n\n"
        current_tokens += para_tokens

    # Don't forget the last chunk
    if current_text.strip():
        chunks.append(FileChunk(
            text=current_text.strip(),
            file_path=file_path,
            file_name=file_name,
            page_number=page_number,
            chunk_index=chunk_index
        ))

    return chunks


def ingest_file(file_path: str) -> list[FileChunk]:
    """Ingest a single file and return chunks."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        console.print(f"[yellow]Unsupported file type: {ext}[/yellow]")
        return []

    console.print(f"[blue]Processing: {path.name}[/blue]")

    all_chunks = []

    if ext == '.pdf':
        pages = extract_text_from_pdf(file_path)
        for page_num, text in pages:
            chunks = chunk_text(text, file_path, page_number=page_num)
            all_chunks.extend(chunks)
    elif ext == '.docx':
        text = extract_text_from_docx(file_path)
        if text:
            all_chunks = chunk_text(text, file_path)
    elif ext in {'.html', '.htm'}:
        text = extract_text_from_html(file_path)
        if text:
            all_chunks = chunk_text(text, file_path)
    else:  # .txt, .md
        text = extract_text_from_txt(file_path)
        if text:
            all_chunks = chunk_text(text, file_path)

    if all_chunks:
        console.print(f"[green]Created {len(all_chunks)} chunks from {path.name}[/green]")

    return all_chunks


def ingest_folder(
    folder_path: str,
    recursive: bool = True,
    extensions: Optional[set[str]] = None,
    save_callback=None
) -> list[FileChunk]:
    """
    Ingest all supported files from a folder.

    Args:
        folder_path: Path to the folder
        recursive: Whether to search subdirectories
        extensions: Set of extensions to include (defaults to all supported)
        save_callback: Optional callback to save chunks after each file

    Returns:
        List of all chunks
    """
    path = Path(folder_path)
    if not path.exists():
        console.print(f"[red]Folder not found: {folder_path}[/red]")
        return []

    exts = extensions or SUPPORTED_EXTENSIONS
    all_chunks = []

    # Find all matching files
    if recursive:
        files = []
        for ext in exts:
            files.extend(path.rglob(f"*{ext}"))
    else:
        files = []
        for ext in exts:
            files.extend(path.glob(f"*{ext}"))

    # Filter out common directories to skip
    skip_dirs = {'node_modules', 'venv', '.venv', '__pycache__', '.git', 'site-packages'}
    files = [f for f in files if not any(skip in str(f) for skip in skip_dirs)]

    # Skip Cora reports (raw ranking data, not useful for RAG)
    cora_patterns = ['_GMW.html', '_GMW.htm', 'goog_', 'Cora_', 'cora_']
    files = [f for f in files if not any(pattern in f.name for pattern in cora_patterns)]

    console.print(f"[blue]Found {len(files)} files to process[/blue]")

    for i, file_path in enumerate(files):
        try:
            chunks = ingest_file(str(file_path))
            all_chunks.extend(chunks)

            # Save incrementally if callback provided
            if chunks and save_callback:
                try:
                    save_callback(chunks)
                    console.print(f"[dim]Saved {len(chunks)} chunks to database[/dim]")
                except Exception as e:
                    console.print(f"[red]Error saving chunks: {e}[/red]")

            # Progress update every 10 files
            if (i + 1) % 10 == 0:
                console.print(f"[dim]Progress: {i + 1}/{len(files)} files processed[/dim]")

        except Exception as e:
            console.print(f"[red]Error processing {file_path}: {e}[/red]")

    console.print(f"[green]Total chunks from folder: {len(all_chunks)}[/green]")
    return all_chunks


def is_local_path(text: str) -> bool:
    """Check if text is a local file or folder path."""
    # Check if it looks like an absolute path
    if text.startswith('/') or text.startswith('~'):
        return True
    # Check if it's a Windows-style path
    if len(text) > 2 and text[1] == ':':
        return True
    return False
