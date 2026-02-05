#!/usr/bin/env python3
"""Script to ingest ScreamingFrog tutorials using Playwright with Oxylabs proxies."""

import os
import sys
import json
import time
import base64
import re
import requests
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, '.')

import tiktoken
from rich.console import Console
from src.embeddings import get_embeddings_manager
from src.ingest import PROXIES  # Import Oxylabs proxies
from config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()

# DataForSEO credentials
DATAFORSEO_USERNAME = "travis@tjgwebdesign.com"
DATAFORSEO_PASSWORD = "cc64de70a1b4aab9"
DATAFORSEO_BASE_URL = "https://api.dataforseo.com"

# ScreamingFrog tutorials base URL
TUTORIALS_BASE_URL = "https://www.screamingfrog.co.uk/seo-spider/tutorials/"

# Current proxy index for rotation
_proxy_index = 0


def get_next_proxy() -> dict:
    """Get next proxy in rotation."""
    global _proxy_index
    proxy = PROXIES[_proxy_index]
    _proxy_index = (_proxy_index + 1) % len(PROXIES)
    return proxy


def get_playwright_proxy(proxy: dict) -> dict:
    """Convert proxy config to Playwright format."""
    return {
        "server": f"http://{proxy['host']}:{proxy['port']}",
        "username": proxy['username'],
        "password": proxy['password'],
    }


@dataclass
class WebPageChunk:
    """A chunk of web page content with metadata."""
    text: str
    url: str
    title: str
    chunk_index: int


def get_auth_header():
    """Get the Basic Auth header for DataForSEO."""
    credentials = f"{DATAFORSEO_USERNAME}:{DATAFORSEO_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def start_onpage_task(target_url: str, max_pages: int = 100) -> str:
    """Start an OnPage crawl task and return the task ID."""
    endpoint = f"{DATAFORSEO_BASE_URL}/v3/on_page/task_post"

    payload = [{
        "target": target_url,
        "max_crawl_pages": max_pages,
        "load_resources": False,
        "enable_javascript": False,
        "enable_browser_rendering": False,
        "store_raw_html": True,
    }]

    response = requests.post(
        endpoint,
        headers=get_auth_header(),
        json=payload
    )

    result = response.json()
    if result.get("status_code") == 20000:
        task_id = result["tasks"][0]["id"]
        console.print(f"[green]Task created: {task_id}[/green]")
        return task_id
    else:
        console.print(f"[red]Error creating task: {result}[/red]")
        return None


def check_task_status(task_id: str) -> dict:
    """Check the status of an OnPage task."""
    endpoint = f"{DATAFORSEO_BASE_URL}/v3/on_page/summary/{task_id}"

    response = requests.get(endpoint, headers=get_auth_header())
    return response.json()


def get_crawled_pages(task_id: str) -> list[dict]:
    """Get all crawled pages from a completed task."""
    endpoint = f"{DATAFORSEO_BASE_URL}/v3/on_page/pages"

    payload = [{
        "id": task_id,
        "limit": 1000,
        "filters": ["resource_type", "=", "html"],
    }]

    response = requests.post(
        endpoint,
        headers=get_auth_header(),
        json=payload
    )

    result = response.json()
    if result.get("status_code") == 20000:
        tasks = result.get("tasks", [])
        if tasks and tasks[0].get("result"):
            return tasks[0]["result"][0].get("items", [])
    return []


def get_page_content(task_id: str, page_url: str) -> dict:
    """Get the raw HTML content of a specific page."""
    endpoint = f"{DATAFORSEO_BASE_URL}/v3/on_page/raw_html"

    payload = [{
        "id": task_id,
        "url": page_url,
    }]

    response = requests.post(
        endpoint,
        headers=get_auth_header(),
        json=payload
    )

    result = response.json()
    if result.get("status_code") == 20000:
        tasks = result.get("tasks", [])
        if tasks and tasks[0].get("result"):
            return tasks[0]["result"][0]
    return None


def extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML."""
    from html.parser import HTMLParser
    from io import StringIO

    class MLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.reset()
            self.strict = False
            self.convert_charrefs = True
            self.text = StringIO()
            self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside'}
            self.current_skip = False

        def handle_starttag(self, tag, attrs):
            if tag in self.skip_tags:
                self.current_skip = True

        def handle_endtag(self, tag):
            if tag in self.skip_tags:
                self.current_skip = False

        def handle_data(self, d):
            if not self.current_skip:
                self.text.write(d + " ")

        def get_data(self):
            return self.text.getvalue()

    s = MLStripper()
    s.feed(html)
    text = s.get_data()

    # Clean up whitespace
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)

    return text


def chunk_text(text: str, url: str, title: str) -> list[WebPageChunk]:
    """Chunk text into smaller pieces."""
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []

    tokens = encoding.encode(text)
    chunk_index = 0

    i = 0
    while i < len(tokens):
        # Get chunk of tokens
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunk_text = encoding.decode(chunk_tokens)

        chunks.append(WebPageChunk(
            text=chunk_text.strip(),
            url=url,
            title=title,
            chunk_index=chunk_index,
        ))

        chunk_index += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def fetch_tutorials_direct() -> list[dict]:
    """Fetch tutorial URLs using DataForSEO SERP API as backup."""
    # Use site: search to find all tutorial pages
    endpoint = f"{DATAFORSEO_BASE_URL}/v3/serp/google/organic/live/advanced"

    payload = [{
        "keyword": "site:screamingfrog.co.uk/seo-spider/tutorials/",
        "location_code": 2840,  # US
        "language_code": "en",
        "depth": 100,
    }]

    response = requests.post(
        endpoint,
        headers=get_auth_header(),
        json=payload
    )

    result = response.json()
    urls = []

    if result.get("status_code") == 20000:
        tasks = result.get("tasks", [])
        if tasks and tasks[0].get("result"):
            for item in tasks[0]["result"]:
                for res in item.get("items", []):
                    if res.get("type") == "organic" and "/tutorials/" in res.get("url", ""):
                        urls.append({
                            "url": res["url"],
                            "title": res.get("title", ""),
                        })

    return urls


def fetch_tutorials_with_playwright(browser) -> list[dict]:
    """Fetch tutorial URLs by scraping the tutorials page directly with Playwright."""
    console.print("[blue]Fetching tutorial links from main page with Playwright...[/blue]")

    tutorials = []
    page = browser.new_page()

    try:
        # Add stealth settings to avoid bot detection
        page.set_extra_http_headers({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })

        # Navigate to main tutorials page
        page.goto(TUTORIALS_BASE_URL, wait_until="networkidle", timeout=60000)

        # Wait for Cloudflare challenge if present
        page.wait_for_timeout(5000)

        # Check if we hit Cloudflare
        content = page.content()
        if "Just a moment" in content or "Checking your browser" in content:
            console.print("[yellow]Cloudflare challenge detected, waiting...[/yellow]")
            page.wait_for_timeout(10000)
            content = page.content()

        # Look for tutorial links - ScreamingFrog uses various selectors
        # Try multiple selectors
        link_selectors = [
            'a[href*="/seo-spider/tutorials/"]',
            '.tutorial-list a',
            'article a',
            '.entry-content a',
            'main a[href*="tutorials"]',
        ]

        found_links = set()

        for selector in link_selectors:
            try:
                links = page.query_selector_all(selector)
                for link in links:
                    href = link.get_attribute('href')
                    title = link.inner_text().strip() or link.get_attribute('title') or ''

                    if href and '/tutorials/' in href:
                        # Make absolute URL
                        if href.startswith('/'):
                            href = f"https://www.screamingfrog.co.uk{href}"

                        # Skip pagination and anchors
                        if '/page/' not in href and '#' not in href and href not in found_links:
                            found_links.add(href)
                            if title and len(title) > 5:  # Only add if title is meaningful
                                tutorials.append({
                                    "url": href,
                                    "title": title[:200],  # Limit title length
                                })
            except Exception as e:
                console.print(f"  [dim]Selector {selector} failed: {e}[/dim]")

        # Also try getting links from the HTML directly using regex as fallback
        import re
        html = page.content()
        pattern = r'href=["\']([^"\']*?/seo-spider/tutorials/[^"\'#]*?)["\']'
        matches = re.findall(pattern, html)

        for match in matches:
            href = match
            if href.startswith('/'):
                href = f"https://www.screamingfrog.co.uk{href}"
            if href not in found_links and '/page/' not in href:
                found_links.add(href)
                tutorials.append({
                    "url": href,
                    "title": "",  # Will extract from page later
                })

        console.print(f"[green]Found {len(tutorials)} tutorial links[/green]")

    except Exception as e:
        console.print(f"[red]Error fetching tutorial links: {e}[/red]")
    finally:
        page.close()

    return tutorials


def fetch_tutorials_with_playwright_context(context, stealth=None) -> list[dict]:
    """Fetch tutorial URLs by scraping the tutorials page with a browser context."""
    console.print("[blue]Fetching tutorial links from main page...[/blue]")

    tutorials = []
    page = context.new_page()

    # Apply stealth to this page if provided
    if stealth:
        stealth.apply_stealth_sync(page)

    try:
        # Navigate to main tutorials page
        console.print(f"  Navigating to {TUTORIALS_BASE_URL}")
        page.goto(TUTORIALS_BASE_URL, wait_until="domcontentloaded", timeout=60000)

        # Wait for page to settle
        page.wait_for_timeout(3000)

        # Check if we hit Cloudflare challenge
        content = page.content()
        if "Just a moment" in content or "Checking your browser" in content or "cf-browser-verification" in content:
            console.print("[yellow]Cloudflare challenge detected, waiting for it to resolve...[/yellow]")
            # Wait longer for Cloudflare to complete
            for _ in range(12):  # Wait up to 60 seconds
                page.wait_for_timeout(5000)
                content = page.content()
                if "Just a moment" not in content and "Checking your browser" not in content:
                    console.print("[green]Cloudflare challenge passed![/green]")
                    break
            else:
                console.print("[red]Cloudflare challenge did not resolve[/red]")
                return []

        # Wait for content to load
        page.wait_for_timeout(2000)

        # Get the page title for debugging
        title = page.title()
        console.print(f"  Page title: {title}")

        # Look for tutorial links
        found_links = {}  # url -> title mapping to avoid duplicates

        # Try multiple selectors
        link_selectors = [
            'a[href*="/seo-spider/tutorials/"]',
            '.tutorial-list a',
            'article a',
            '.entry-content a',
            'main a[href*="tutorials"]',
            '.post a',
            'h2 a',
            'h3 a',
        ]

        for selector in link_selectors:
            try:
                links = page.query_selector_all(selector)
                for link in links:
                    href = link.get_attribute('href')
                    link_title = link.inner_text().strip() or link.get_attribute('title') or ''

                    if href and '/tutorials/' in href:
                        # Make absolute URL
                        if href.startswith('/'):
                            href = f"https://www.screamingfrog.co.uk{href}"

                        # Skip pagination, anchors, and the main tutorials page itself
                        if '/page/' not in href and '#' not in href:
                            # Don't add if we already have this URL with a title
                            if href not in found_links or (not found_links[href] and link_title):
                                found_links[href] = link_title
            except Exception as e:
                pass  # Silently continue if selector fails

        # Also extract links via regex from HTML
        import re
        html = page.content()
        pattern = r'href=["\']([^"\']*?/seo-spider/tutorials/[^"\'#]*?)["\']'
        matches = re.findall(pattern, html)

        for match in matches:
            href = match
            if href.startswith('/'):
                href = f"https://www.screamingfrog.co.uk{href}"
            if href not in found_links and '/page/' not in href:
                found_links[href] = ""

        # Convert to list format
        for url, title in found_links.items():
            # Skip the main tutorials listing page
            if url.rstrip('/') == TUTORIALS_BASE_URL.rstrip('/'):
                continue
            tutorials.append({
                "url": url,
                "title": title if title and len(title) > 3 else "ScreamingFrog Tutorial",
            })

        console.print(f"[green]Found {len(tutorials)} tutorial links[/green]")

        # Print first few for verification
        if tutorials:
            console.print("[dim]Sample tutorials found:[/dim]")
            for t in tutorials[:5]:
                console.print(f"  [dim]- {t['title']}: {t['url']}[/dim]")

    except Exception as e:
        console.print(f"[red]Error fetching tutorial links: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        page.close()

    return tutorials


def fetch_page_html_with_context(url: str, context, stealth=None) -> tuple[str, str]:
    """Fetch page content using a browser context. Returns (html, title)."""
    page = context.new_page()

    # Apply stealth if provided
    if stealth:
        stealth.apply_stealth_sync(page)

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # Wait for content
        page.wait_for_timeout(2000)

        # Check for Cloudflare
        content = page.content()
        if "Just a moment" in content or "Checking your browser" in content:
            console.print("    [yellow]Waiting for Cloudflare...[/yellow]")
            for _ in range(6):
                page.wait_for_timeout(5000)
                content = page.content()
                if "Just a moment" not in content:
                    break

        title = page.title()
        html = page.content()
        return html, title
    except Exception as e:
        console.print(f"    [dim]Fetch failed: {e}[/dim]")
        return None, None
    finally:
        page.close()


def fetch_page_content_direct(url: str) -> dict:
    """Fetch a single page's content using DataForSEO Content Analysis API."""
    endpoint = f"{DATAFORSEO_BASE_URL}/v3/content_analysis/summary/live"

    payload = [{
        "page_url": url,
    }]

    response = requests.post(
        endpoint,
        headers=get_auth_header(),
        json=payload
    )

    result = response.json()
    if result.get("status_code") == 20000:
        tasks = result.get("tasks", [])
        if tasks and tasks[0].get("result"):
            items = tasks[0]["result"]
            if items:
                return items[0]
    return None


def fetch_page_html_playwright(url: str, browser) -> str:
    """Fetch page content using Playwright browser."""
    try:
        page = browser.new_page()
        # Navigate and wait for load
        page.goto(url, wait_until="load", timeout=60000)

        # Try to wait for the main content area to appear
        try:
            page.wait_for_selector("article, .entry-content, .post-content, main", timeout=10000)
        except:
            # If no content selector found, just wait longer
            page.wait_for_timeout(5000)

        # Additional wait for JS rendering
        page.wait_for_timeout(2000)

        html = page.content()
        page.close()
        return html
    except Exception as e:
        console.print(f"    [dim]Playwright fetch failed: {e}[/dim]")
        try:
            page.close()
        except:
            pass
        return None


def main():
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    manager = get_embeddings_manager()

    # Show starting stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Starting DB stats:[/bold]")
    console.print(f"  Total: {stats['total']}")

    all_chunks = []

    # Get a proxy for the browser
    proxy = get_next_proxy()
    playwright_proxy = get_playwright_proxy(proxy)
    console.print(f"\n[blue]Using Oxylabs proxy: {proxy['host']}:{proxy['port']}[/blue]")

    # Create stealth instance with Mac-appropriate settings
    stealth = Stealth(
        navigator_platform_override='MacIntel',
        navigator_vendor_override='Google Inc.',
    )

    # Start Playwright browser with proxy and stealth settings
    console.print(f"[blue]Starting browser with proxy and stealth mode...[/blue]")
    with sync_playwright() as p:
        # Launch browser - use headed mode for better Cloudflare bypass
        browser = p.chromium.launch(
            headless=False,  # Headed mode helps bypass Cloudflare
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )

        # Create context with proxy and realistic settings
        context = browser.new_context(
            proxy=playwright_proxy,
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
        )

        # Fetch tutorial URLs using Playwright with proxy and stealth
        console.print(f"\n[blue]Fetching ScreamingFrog tutorial URLs with Playwright + Proxy + Stealth...[/blue]")
        tutorials = fetch_tutorials_with_playwright_context(context, stealth)

        if not tutorials:
            console.print("[yellow]Playwright extraction failed, trying DataForSEO SERP...[/yellow]")
            tutorials = fetch_tutorials_direct()
            console.print(f"[green]Found {len(tutorials)} tutorial pages from SERP[/green]")

        if not tutorials:
            console.print("[yellow]No tutorials found, trying manual list...[/yellow]")
            # Fallback to known tutorial URLs
            tutorials = [
                {"url": "https://www.screamingfrog.co.uk/seo-spider/tutorials/", "title": "SEO Spider Tutorials"},
            ]

        # Filter to only actual tutorial pages (not pagination)
        tutorials = [t for t in tutorials if not t["url"].endswith("/page/2/") and "/tutorials/" in t["url"]]
        console.print(f"[blue]Filtered to {len(tutorials)} actual tutorial pages[/blue]")

        for i, tutorial in enumerate(tutorials):
            url = tutorial["url"]
            title = tutorial.get("title", "ScreamingFrog Tutorial")

            console.print(f"\n[bold]Processing {i+1}/{len(tutorials)}:[/bold] {title}")
            console.print(f"  URL: {url}")

            try:
                # Fetch page HTML with Playwright using the same context (shares cookies/session)
                html, page_title = fetch_page_html_with_context(url, context, stealth)

                if not html:
                    console.print(f"  [yellow]Could not fetch page[/yellow]")
                    continue

                # Use page title if we don't have one
                if title == "ScreamingFrog Tutorial" and page_title:
                    title = page_title

                # Extract text
                text = extract_text_from_html(html)
                if len(text) < 100:
                    console.print(f"  [yellow]Too little content ({len(text)} chars)[/yellow]")
                    continue

                console.print(f"  [dim]Extracted {len(text)} characters[/dim]")

                # Chunk the text
                chunks = chunk_text(text, url, title)
                console.print(f"  [dim]Created {len(chunks)} chunks[/dim]")

                all_chunks.extend(chunks)

                # Rate limiting - be gentle to avoid getting blocked
                time.sleep(1.0)

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                import traceback
                traceback.print_exc()

        context.close()
        browser.close()

    # Convert to nodes and add to database
    if all_chunks:
        from llama_index.core.schema import TextNode

        nodes = []
        for chunk in all_chunks:
            node = TextNode(
                text=chunk.text,
                metadata={
                    'source_type': 'web',
                    'url': chunk.url,
                    'page_title': chunk.title,
                    'site_name': 'ScreamingFrog',
                    'chunk_index': chunk.chunk_index,
                },
                excluded_embed_metadata_keys=['url', 'chunk_index', 'source_type'],
                excluded_llm_metadata_keys=['chunk_index', 'source_type'],
            )
            nodes.append(node)

        console.print(f"\n[blue]Adding {len(nodes)} chunks to database...[/blue]")

        from llama_index.core import VectorStoreIndex

        manager._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=manager.storage_context,
            show_progress=True
        )

        console.print(f"[green]Added {len(nodes)} chunks[/green]")

    # Show final stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Final DB stats:[/bold]")
    console.print(f"  Total: {stats['total']}")
    console.print(f"\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
