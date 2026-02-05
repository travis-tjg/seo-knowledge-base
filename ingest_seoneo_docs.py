#!/usr/bin/env python3
"""Ingest SEO Neo documentation from docs.seoneo.io"""

import sys
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.embeddings import get_embeddings_manager
from src.file_ingest import FileChunk


def get_all_doc_urls(base_url: str = "https://docs.seoneo.io/") -> list[str]:
    """Crawl the docs site and get all documentation page URLs."""
    visited = set()
    to_visit = [base_url]
    doc_urls = []

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            # This is a doc page, add it
            if urlparse(url).netloc == "docs.seoneo.io":
                doc_urls.append(url)
                print(f"Found: {url}")

            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)

                # Only follow links within docs.seoneo.io
                parsed = urlparse(full_url)
                if parsed.netloc == "docs.seoneo.io" and full_url not in visited:
                    # Clean up URL (remove fragments)
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    if clean_url not in visited and clean_url not in to_visit:
                        to_visit.append(clean_url)

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            continue

    return list(set(doc_urls))


def extract_page_content(url: str) -> tuple[str, str]:
    """Extract title and main content from a documentation page."""
    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else url

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            content = main_content.get_text(separator='\n', strip=True)
        else:
            content = soup.get_text(separator='\n', strip=True)

        # Clean up content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        content = '\n'.join(lines)

        return title_text, content

    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return "", ""


def ingest_seoneo_docs():
    """Main function to ingest SEO Neo documentation."""
    print("Crawling docs.seoneo.io for documentation pages...")
    doc_urls = get_all_doc_urls()
    print(f"\nFound {len(doc_urls)} documentation pages")

    if not doc_urls:
        print("No pages found!")
        return

    manager = get_embeddings_manager()
    chunks = []

    for i, url in enumerate(doc_urls):
        title, content = extract_page_content(url)

        if not content or len(content) < 100:
            print(f"Skipping {url} - too short")
            continue

        # Create a chunk for this page
        chunk = FileChunk(
            text=f"Title: {title}\n\n{content}",
            file_path=url,
            file_name=title,
            page_number=1,
            chunk_index=i
        )
        chunks.append(chunk)
        print(f"[{i+1}/{len(doc_urls)}] Processed: {title[:50]}...")

    if chunks:
        print(f"\nAdding {len(chunks)} documentation pages to vector store...")
        manager.add_chunks(chunks)
        print("Done!")

    # Get final stats
    stats = manager.get_stats()
    print(f"\nDatabase stats: {stats}")


if __name__ == "__main__":
    ingest_seoneo_docs()
