"""Web page ingestion pipeline for scraping and chunking web content."""

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests
import tiktoken
from bs4 import BeautifulSoup
from rich.console import Console

import sys
sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
from config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()

# User agent to avoid being blocked
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


@dataclass
class WebChunk:
    """A chunk of web content with metadata."""
    text: str
    url: str
    title: str
    chunk_index: int
    domain: Optional[str] = None


def is_web_url(text: str) -> bool:
    """Check if text is a web URL (not YouTube)."""
    text = text.strip()
    if not text.startswith(('http://', 'https://')):
        return False
    # Exclude YouTube URLs (handled by ingest.py)
    parsed = urlparse(text)
    youtube_domains = {'youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com'}
    if parsed.netloc in youtube_domains:
        return False
    return True


def fetch_web_page(url: str, timeout: int = 30) -> tuple[str, str]:
    """
    Fetch a web page and return (html_content, final_url).
    Follows redirects and handles common issues.
    """
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()

    return response.text, response.url


def extract_text_from_html(html: str, url: str) -> tuple[str, str]:
    """
    Extract clean text from HTML content.
    Returns (text, title).
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Get title
    title = ""
    if soup.title:
        title = soup.title.string or ""
    if not title:
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        title = urlparse(url).netloc

    # Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer',
                     'iframe', 'noscript', 'aside', 'form']):
        tag.decompose()

    # Remove elements that are typically not content
    for element in soup.find_all(class_=re.compile(r'nav|menu|sidebar|footer|header|cookie|banner|ad-|advertisement', re.I)):
        element.decompose()

    # Try to find main content area
    main_content = None
    for selector in ['main', 'article', '[role="main"]', '.content', '.post-content',
                     '.entry-content', '.article-content', '#content', '#main']:
        main_content = soup.select_one(selector)
        if main_content:
            break

    if main_content:
        text = main_content.get_text(separator='\n')
    else:
        # Fall back to body
        body = soup.find('body')
        text = body.get_text(separator='\n') if body else soup.get_text(separator='\n')

    # Clean up whitespace
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    clean_text = '\n'.join(lines)

    # Collapse multiple newlines
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    return clean_text, title


def chunk_web_content(
    text: str,
    url: str,
    title: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_tokens: int = 7000
) -> list[WebChunk]:
    """Chunk web content into pieces of approximately chunk_size tokens."""
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []
    domain = urlparse(url).netloc

    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)

    current_text = ""
    current_tokens = 0
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = len(encoding.encode(para + "\n\n"))

        # If paragraph itself is too large, split by sentences
        if para_tokens > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_tokens = len(encoding.encode(sent + " "))

                # Handle very long sentences by splitting on words
                if sent_tokens > max_tokens:
                    words = sent.split()
                    word_chunk = ""
                    word_tokens = 0
                    for word in words:
                        wt = len(encoding.encode(word + " "))
                        if word_tokens + wt > max_tokens and word_chunk:
                            if current_text:
                                chunks.append(WebChunk(
                                    text=current_text.strip(),
                                    url=url,
                                    title=title,
                                    chunk_index=chunk_index,
                                    domain=domain
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
                    chunks.append(WebChunk(
                        text=current_text.strip(),
                        url=url,
                        title=title,
                        chunk_index=chunk_index,
                        domain=domain
                    ))
                    chunk_index += 1
                    current_text = ""
                    current_tokens = 0

                current_text += sent + " "
                current_tokens += sent_tokens
            continue

        # If adding this paragraph exceeds chunk size, save current chunk
        if current_tokens + para_tokens > chunk_size and current_text:
            chunks.append(WebChunk(
                text=current_text.strip(),
                url=url,
                title=title,
                chunk_index=chunk_index,
                domain=domain
            ))
            chunk_index += 1

            # Keep overlap
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
        chunks.append(WebChunk(
            text=current_text.strip(),
            url=url,
            title=title,
            chunk_index=chunk_index,
            domain=domain
        ))

    return chunks


def ingest_web_page(url: str) -> list[WebChunk]:
    """
    Ingest a web page and return chunks.

    Args:
        url: URL of the web page to ingest

    Returns:
        List of WebChunk objects
    """
    console.print(f"[blue]Fetching: {url}[/blue]")

    try:
        html, final_url = fetch_web_page(url)

        if final_url != url:
            console.print(f"[dim]Redirected to: {final_url}[/dim]")

        text, title = extract_text_from_html(html, final_url)

        if not text or len(text.strip()) < 100:
            console.print("[yellow]Warning: Very little text content extracted from page[/yellow]")
            if not text.strip():
                return []

        console.print(f"[dim]Title: {title}[/dim]")
        console.print(f"[dim]Extracted {len(text)} characters of text[/dim]")

        chunks = chunk_web_content(text, final_url, title)

        if chunks:
            console.print(f"[green]Created {len(chunks)} chunks from web page[/green]")

        return chunks

    except requests.exceptions.Timeout:
        console.print(f"[red]Timeout fetching URL: {url}[/red]")
        return []
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error fetching URL: {e}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]Error processing web page: {e}[/red]")
        import traceback
        traceback.print_exc()
        return []
