#!/usr/bin/env python3
"""
Script to scrape ScreamingFrog tutorials with manual CAPTCHA solving.

This script opens a visible browser window where you can manually solve
the Cloudflare CAPTCHA. Once solved, it saves the cookies and proceeds
to scrape all tutorial pages.

Usage:
    python scrape_screamingfrog_manual.py
"""

import os
import sys
import json
import time
import re
from pathlib import Path

sys.path.insert(0, '.')

import tiktoken
from rich.console import Console
from rich.prompt import Confirm
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

from src.embeddings import get_embeddings_manager
from src.ingest import PROXIES  # Import Oxylabs proxies
from config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()

TUTORIALS_BASE_URL = "https://www.screamingfrog.co.uk/seo-spider/tutorials/"
COOKIES_FILE = Path("data/screamingfrog_cookies.json")


def get_playwright_proxy(proxy: dict) -> dict:
    """Convert proxy config to Playwright format."""
    return {
        "server": f"http://{proxy['host']}:{proxy['port']}",
        "username": proxy['username'],
        "password": proxy['password'],
    }


def save_cookies(context, filepath: Path):
    """Save browser cookies to a file."""
    cookies = context.cookies()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(cookies, f, indent=2)
    console.print(f"[green]Saved {len(cookies)} cookies to {filepath}[/green]")


def load_cookies(context, filepath: Path) -> bool:
    """Load cookies from file into browser context."""
    if not filepath.exists():
        return False
    try:
        with open(filepath, 'r') as f:
            cookies = json.load(f)
        context.add_cookies(cookies)
        console.print(f"[green]Loaded {len(cookies)} cookies from {filepath}[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]Could not load cookies: {e}[/yellow]")
        return False


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
            self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript'}
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


def chunk_text(text: str, url: str, title: str) -> list:
    """Chunk text into smaller pieces."""
    from dataclasses import dataclass

    @dataclass
    class WebPageChunk:
        text: str
        url: str
        title: str
        chunk_index: int

    encoding = tiktoken.get_encoding("cl100k_base")
    chunks = []

    tokens = encoding.encode(text)
    chunk_index = 0

    i = 0
    while i < len(tokens):
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


def is_cloudflare_challenge(content: str) -> bool:
    """Check if page is showing Cloudflare challenge."""
    indicators = [
        "Just a moment",
        "Checking your browser",
        "cf-browser-verification",
        "Enable JavaScript and cookies to continue",
        "Verify you are human",
    ]
    return any(ind in content for ind in indicators)


def wait_for_manual_captcha(page, timeout_minutes: int = 5):
    """Wait for user to solve CAPTCHA manually."""
    console.print("\n[bold yellow]" + "="*60 + "[/bold yellow]")
    console.print("[bold yellow]CLOUDFLARE CAPTCHA DETECTED![/bold yellow]")
    console.print("[bold yellow]" + "="*60 + "[/bold yellow]")
    console.print("\n[cyan]Please solve the CAPTCHA in the browser window.[/cyan]")
    console.print(f"[cyan]Waiting up to {timeout_minutes} minutes...[/cyan]\n")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start_time < timeout_seconds:
        content = page.content()
        if not is_cloudflare_challenge(content):
            console.print("[bold green]CAPTCHA solved! Continuing...[/bold green]\n")
            return True
        time.sleep(2)

    console.print("[bold red]Timeout waiting for CAPTCHA to be solved.[/bold red]")
    return False


def extract_tutorial_links(page) -> list[dict]:
    """Extract all tutorial links from the current page."""
    tutorials = []
    found_urls = set()

    # Get HTML and extract links via regex (more reliable than selectors)
    html = page.content()
    pattern = r'href=["\']([^"\']*?/seo-spider/tutorials/[^"\'#]*?)["\']'
    matches = re.findall(pattern, html)

    for href in matches:
        if href.startswith('/'):
            href = f"https://www.screamingfrog.co.uk{href}"

        # Skip pagination, main page, and duplicates
        if '/page/' in href:
            continue
        if href.rstrip('/') == TUTORIALS_BASE_URL.rstrip('/'):
            continue
        if href in found_urls:
            continue

        found_urls.add(href)
        tutorials.append({"url": href, "title": ""})

    # Try to get titles from links
    try:
        links = page.query_selector_all('a[href*="/seo-spider/tutorials/"]')
        for link in links:
            href = link.get_attribute('href')
            if href and href.startswith('/'):
                href = f"https://www.screamingfrog.co.uk{href}"

            title = link.inner_text().strip()
            if href and title and len(title) > 5:
                # Update title for existing URL
                for t in tutorials:
                    if t['url'] == href:
                        t['title'] = title[:200]
                        break
    except:
        pass

    return tutorials


def main():
    manager = get_embeddings_manager()

    # Show starting stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Starting DB stats:[/bold]")
    console.print(f"  Total: {stats['total']}")

    all_chunks = []

    # Get proxy
    proxy = PROXIES[0]  # Use first proxy
    playwright_proxy = get_playwright_proxy(proxy)
    console.print(f"\n[blue]Using proxy: {proxy['host']}:{proxy['port']}[/blue]")

    # Create stealth instance
    stealth = Stealth(
        navigator_platform_override='MacIntel',
        navigator_vendor_override='Google Inc.',
    )

    console.print(f"[blue]Starting browser (visible mode for CAPTCHA solving)...[/blue]")

    with sync_playwright() as p:
        # Launch visible browser
        browser = p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
            ]
        )

        context = browser.new_context(
            proxy=playwright_proxy,
            viewport={'width': 1400, 'height': 900},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
        )

        # Try to load existing cookies
        cookies_loaded = load_cookies(context, COOKIES_FILE)

        page = context.new_page()
        stealth.apply_stealth_sync(page)

        # Navigate to tutorials page
        console.print(f"\n[blue]Navigating to {TUTORIALS_BASE_URL}[/blue]")
        page.goto(TUTORIALS_BASE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        # Check for Cloudflare
        if is_cloudflare_challenge(page.content()):
            if not wait_for_manual_captcha(page, timeout_minutes=5):
                console.print("[red]Could not bypass Cloudflare. Exiting.[/red]")
                browser.close()
                return

            # Save cookies after solving CAPTCHA
            save_cookies(context, COOKIES_FILE)

        # Extract tutorial links
        console.print("\n[blue]Extracting tutorial links...[/blue]")
        tutorials = extract_tutorial_links(page)
        console.print(f"[green]Found {len(tutorials)} tutorial pages[/green]")

        if not tutorials:
            console.print("[yellow]No tutorials found on the page.[/yellow]")
            browser.close()
            return

        # Show sample
        console.print("\n[dim]Sample tutorials:[/dim]")
        for t in tutorials[:5]:
            console.print(f"  [dim]- {t['url']}[/dim]")

        page.close()

        # Process each tutorial
        for i, tutorial in enumerate(tutorials):
            url = tutorial["url"]
            title = tutorial.get("title", "")

            console.print(f"\n[bold]Processing {i+1}/{len(tutorials)}:[/bold] {url}")

            try:
                page = context.new_page()
                stealth.apply_stealth_sync(page)
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(2000)

                # Check for Cloudflare again
                if is_cloudflare_challenge(page.content()):
                    console.print("  [yellow]Cloudflare challenge on this page...[/yellow]")
                    if not wait_for_manual_captcha(page, timeout_minutes=3):
                        console.print("  [red]Skipping page[/red]")
                        page.close()
                        continue
                    save_cookies(context, COOKIES_FILE)

                # Get page title if we don't have one
                if not title:
                    title = page.title() or "ScreamingFrog Tutorial"

                html = page.content()
                page.close()

                # Extract text
                text = extract_text_from_html(html)
                if len(text) < 200:
                    console.print(f"  [yellow]Too little content ({len(text)} chars), skipping[/yellow]")
                    continue

                console.print(f"  [dim]Extracted {len(text)} characters[/dim]")

                # Chunk the text
                chunks = chunk_text(text, url, title)
                console.print(f"  [dim]Created {len(chunks)} chunks[/dim]")

                all_chunks.extend(chunks)

                # Rate limiting
                time.sleep(1.5)

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                try:
                    page.close()
                except:
                    pass

        context.close()
        browser.close()

    # Convert to nodes and add to database
    if all_chunks:
        from llama_index.core.schema import TextNode
        from llama_index.core import VectorStoreIndex

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

        manager._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=manager.storage_context,
            show_progress=True
        )

        console.print(f"[green]Added {len(nodes)} chunks[/green]")
    else:
        console.print("[yellow]No chunks to add to database.[/yellow]")

    # Show final stats
    stats = manager.get_stats_by_source()
    console.print(f"\n[bold]Final DB stats:[/bold]")
    console.print(f"  Total: {stats['total']}")
    console.print(f"\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
