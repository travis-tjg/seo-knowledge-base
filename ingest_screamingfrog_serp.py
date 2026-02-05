#!/usr/bin/env python3
"""
Ingest ScreamingFrog tutorials using DataForSEO SERP data.

Since Cloudflare blocks direct scraping, this script uses Google's indexed
content (titles and descriptions) from DataForSEO SERP API to create
searchable entries for each tutorial.
"""

import sys
import base64
import requests

sys.path.insert(0, '.')

from rich.console import Console
from src.embeddings import get_embeddings_manager

console = Console()

# DataForSEO credentials
DATAFORSEO_USERNAME = "travis@tjgwebdesign.com"
DATAFORSEO_PASSWORD = "cc64de70a1b4aab9"
DATAFORSEO_BASE_URL = "https://api.dataforseo.com"


def get_auth_header():
    """Get the Basic Auth header for DataForSEO."""
    credentials = f"{DATAFORSEO_USERNAME}:{DATAFORSEO_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def fetch_all_tutorials() -> list[dict]:
    """Fetch all tutorial data from SERP."""
    console.print("[blue]Fetching tutorial data from DataForSEO SERP...[/blue]")

    all_tutorials = []

    # Search for tutorials with different queries to get more coverage
    search_queries = [
        "site:screamingfrog.co.uk/seo-spider/tutorials/",
        "site:screamingfrog.co.uk seo spider tutorial how to",
        "site:screamingfrog.co.uk seo spider guide crawl",
    ]

    seen_urls = set()

    for query in search_queries:
        endpoint = f"{DATAFORSEO_BASE_URL}/v3/serp/google/organic/live/advanced"

        payload = [{
            "keyword": query,
            "location_code": 2840,  # US
            "language_code": "en",
            "depth": 100,
        }]

        response = requests.post(endpoint, headers=get_auth_header(), json=payload)
        result = response.json()

        if result.get("status_code") == 20000:
            tasks = result.get("tasks", [])
            if tasks and tasks[0].get("result"):
                for item in tasks[0]["result"]:
                    for res in item.get("items", []):
                        if res.get("type") == "organic":
                            url = res.get("url", "")
                            # Only include tutorial pages
                            if "/tutorials/" in url and url not in seen_urls:
                                # Skip pagination pages
                                if "/page/" in url:
                                    continue

                                seen_urls.add(url)
                                all_tutorials.append({
                                    "url": url,
                                    "title": res.get("title", ""),
                                    "description": res.get("description", ""),
                                })

    console.print(f"[green]Found {len(all_tutorials)} unique tutorials[/green]")
    return all_tutorials


def create_tutorial_text(tutorial: dict) -> str:
    """Create searchable text from tutorial data."""
    parts = []

    # Add title
    if tutorial.get("title"):
        parts.append(f"# {tutorial['title']}")

    # Add URL for reference
    parts.append(f"\nSource: {tutorial['url']}")

    # Add description
    if tutorial.get("description"):
        parts.append(f"\n{tutorial['description']}")

    # Add category hint from URL
    url = tutorial.get("url", "")
    if "/tutorials/" in url:
        slug = url.split("/tutorials/")[-1].rstrip("/")
        if slug:
            # Convert slug to readable text
            readable = slug.replace("-", " ").title()
            parts.append(f"\nTopic: {readable}")

    return "\n".join(parts)


def main():
    manager = get_embeddings_manager()

    # Show starting stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Starting DB stats:[/bold]")
    console.print(f"  Total: {stats['total']}")

    # Fetch tutorials from SERP
    tutorials = fetch_all_tutorials()

    if not tutorials:
        console.print("[red]No tutorials found![/red]")
        return

    # Filter out the main listing page
    tutorials = [t for t in tutorials if not t["url"].rstrip("/").endswith("/tutorials")]
    console.print(f"[blue]Processing {len(tutorials)} tutorial pages[/blue]")

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
                'content_source': 'serp_index',  # Mark as coming from SERP
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

    console.print("\n[yellow]Note: This ingested SERP data (titles/descriptions) since Cloudflare blocks direct page scraping. For full content, you may need to use a premium web scraping service or manually access the pages.[/yellow]")


if __name__ == "__main__":
    main()
