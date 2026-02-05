#!/usr/bin/env python3
"""
Ingest ScreamingFrog tutorials using Oxylabs Google SERP API.

Uses Oxylabs to fetch Google search results which contain the indexed
content (titles, descriptions, and snippets) for each tutorial.
"""

import os
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, '.')

from dotenv import load_dotenv
from rich.console import Console
from src.embeddings import get_embeddings_manager

# Load .env file
load_dotenv(Path(__file__).parent / '.env')

console = Console()

# Oxylabs credentials - default to hardcoded if env not set
OXYLABS_USER = os.environ.get('OXYLAB_USER', 'travis_jo_KVoPm')
OXYLABS_PASS = os.environ.get('OXYLAB_PASSWORD', '=G007dec2026')


def search_google(query: str) -> list[dict]:
    """Search Google via Oxylabs and return parsed results."""
    payload = {
        "source": "google_search",
        "query": query,
        "geo_location": "United States",
        "parse": True,
        "pages": 1,
    }

    response = requests.post(
        "https://realtime.oxylabs.io/v1/queries",
        auth=(OXYLABS_USER, OXYLABS_PASS),
        json=payload,
        timeout=120
    )

    if response.status_code == 200:
        result = response.json()
        if result.get('results'):
            content = result['results'][0].get('content', {})
            if isinstance(content, dict):
                return content.get('results', {}).get('organic', [])
    return []


def fetch_all_tutorials() -> list[dict]:
    """Fetch all tutorial data using multiple Google searches."""
    console.print("[blue]Fetching tutorials via Oxylabs Google SERP...[/blue]")

    all_tutorials = {}  # url -> data

    # Different search queries to maximize coverage
    search_queries = [
        "site:screamingfrog.co.uk/seo-spider/tutorials/",
        "site:screamingfrog.co.uk seo spider tutorial how to",
        "site:screamingfrog.co.uk seo spider guide crawl audit",
        "site:screamingfrog.co.uk seo spider javascript rendering",
        "site:screamingfrog.co.uk seo spider structured data",
        "site:screamingfrog.co.uk seo spider hreflang",
        "site:screamingfrog.co.uk seo spider sitemap",
        "site:screamingfrog.co.uk seo spider broken links",
        "site:screamingfrog.co.uk seo spider redirects migration",
        "site:screamingfrog.co.uk seo spider canonicals",
        "site:screamingfrog.co.uk seo spider core web vitals",
        "site:screamingfrog.co.uk seo spider accessibility",
    ]

    for i, query in enumerate(search_queries):
        console.print(f"  [{i+1}/{len(search_queries)}] Searching: {query[:50]}...")

        results = search_google(query)

        for item in results:
            url = item.get('url', '')
            # Only include tutorial pages
            if '/tutorials/' not in url:
                continue
            # Skip pagination pages
            if '/page/' in url:
                continue
            # Skip the main listing page
            if url.rstrip('/').endswith('/tutorials'):
                continue

            if url not in all_tutorials:
                all_tutorials[url] = {
                    'url': url,
                    'title': item.get('title', ''),
                    'description': item.get('desc', ''),
                }
            else:
                # Update with longer description if available
                existing = all_tutorials[url]
                new_desc = item.get('desc', '')
                if len(new_desc) > len(existing.get('description', '')):
                    existing['description'] = new_desc

        # Rate limiting
        time.sleep(1)

    tutorials = list(all_tutorials.values())
    console.print(f"[green]Found {len(tutorials)} unique tutorials[/green]")
    return tutorials


def create_tutorial_text(tutorial: dict) -> str:
    """Create searchable text from tutorial data."""
    parts = []

    # Add title
    if tutorial.get('title'):
        parts.append(f"# {tutorial['title']}")

    # Add URL for reference
    parts.append(f"\nSource: {tutorial['url']}")

    # Add description (the most valuable content we have)
    if tutorial.get('description'):
        parts.append(f"\n{tutorial['description']}")

    # Extract topic hints from URL
    url = tutorial.get('url', '')
    if '/tutorials/' in url:
        slug = url.split('/tutorials/')[-1].rstrip('/')
        if slug:
            # Convert slug to keywords
            keywords = slug.replace('-', ' ').replace('/', ' ').strip()
            if keywords:
                parts.append(f"\nKeywords: {keywords}")

    return "\n".join(parts)


def main():
    manager = get_embeddings_manager()

    # Show starting stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Starting DB stats:[/bold]")
    console.print(f"  Total: {stats['total']}")

    # First, remove old ScreamingFrog SERP entries if any (to avoid duplicates)
    # We'll just add new ones - duplicates will be handled by the vector store

    # Fetch tutorials
    tutorials = fetch_all_tutorials()

    if not tutorials:
        console.print("[red]No tutorials found![/red]")
        return

    console.print(f"\n[blue]Processing {len(tutorials)} tutorials...[/blue]")

    # Show some examples
    console.print("\n[dim]Sample tutorials:[/dim]")
    for t in tutorials[:5]:
        console.print(f"  [dim]- {t['title']}[/dim]")

    # Create nodes
    from llama_index.core.schema import TextNode

    nodes = []
    for i, tutorial in enumerate(tutorials):
        text = create_tutorial_text(tutorial)

        console.print(f"  [{i+1}/{len(tutorials)}] {tutorial['title'][:50]}...")

        node = TextNode(
            text=text,
            metadata={
                'source_type': 'web',
                'url': tutorial['url'],
                'page_title': tutorial['title'],
                'site_name': 'ScreamingFrog',
                'content_source': 'google_serp_oxylabs',
            },
            excluded_embed_metadata_keys=['url', 'content_source', 'source_type'],
            excluded_llm_metadata_keys=['content_source', 'source_type'],
        )
        nodes.append(node)

    if nodes:
        console.print(f"\n[blue]Adding {len(nodes)} tutorial entries to database...[/blue]")

        from llama_index.core import VectorStoreIndex

        manager._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=manager.storage_context,
            show_progress=True
        )

        console.print(f"[green]Added {len(nodes)} entries[/green]")

    # Show final stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Final DB stats:[/bold]")
    console.print(f"  Total: {stats['total']}")
    console.print(f"\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
