"""CLI interface for SEO Knowledge Base."""

import os
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
from src.ingest import ingest_url, is_youtube_url
from src.file_ingest import ingest_folder, ingest_file, is_local_path, SUPPORTED_EXTENSIONS
from src.video_ingest import ingest_video, ingest_video_folder, VIDEO_EXTENSIONS
from src.web_ingest import ingest_web_page, is_web_url
from src.embeddings import get_embeddings_manager
from src.query import query
from config import DOMAINS

console = Console()


def print_welcome():
    """Print welcome message and instructions."""
    console.print(Panel.fit(
        "[bold blue]Multi-Domain Knowledge Base[/bold blue]\n\n"
        "A RAG system for SEO and AI Website Building knowledge.\n\n"
        "[dim]Query Commands:[/dim]\n"
        "  • Type a question to query (auto-routes to best domain)\n"
        "  • [bold]@seo[/bold] <question> - Query SEO knowledge only\n"
        "  • [bold]@web[/bold] <question> - Query AI Website Building only\n"
        "  • [bold]@all[/bold] <question> - Query both domains\n\n"
        "[dim]Ingestion Commands:[/dim]\n"
        "  • [bold]ingest seo[/bold] <url/path> - Add content to SEO KB\n"
        "  • [bold]ingest web[/bold] <url/path> - Add content to Web Builder KB\n"
        "  • [bold]transcribe seo[/bold] <path> - Transcribe video to SEO KB\n"
        "  • [bold]transcribe web[/bold] <path> - Transcribe video to Web Builder KB\n\n"
        "[dim]Other Commands:[/dim]\n"
        "  • [bold]stats[/bold] - Show statistics for all knowledge bases\n"
        "  • [bold]clear seo[/bold] or [bold]clear web[/bold] - Clear a specific KB\n"
        "  • [bold]quit[/bold] or [bold]exit[/bold] to exit",
        title="Welcome",
        border_style="blue"
    ))
    console.print()


def print_stats():
    """Print statistics about all knowledge bases."""
    stats_lines = []

    for domain_name, domain_config in DOMAINS.items():
        manager = get_embeddings_manager(domain_name)
        stats = manager.get_stats()
        source_stats = manager.get_stats_by_source()

        stats_lines.append(f"[bold cyan]{domain_config['display_name']}[/bold cyan]")
        stats_lines.append(f"  Collection: {stats['collection_name']}")
        stats_lines.append(f"  Total chunks: {stats['total_chunks']}")
        if stats['total_chunks'] > 0:
            stats_lines.append(f"    - YouTube: {source_stats.get('youtube', 0)}")
            stats_lines.append(f"    - Local files: {source_stats.get('file', 0)}")
            stats_lines.append(f"    - Local videos: {source_stats.get('local_video', 0)}")
            stats_lines.append(f"    - Web pages: {source_stats.get('web', 0)}")
        stats_lines.append("")

    stats_lines.append(f"[dim]Embedding model: text-embedding-3-small[/dim]")

    console.print(Panel.fit(
        "\n".join(stats_lines),
        title="Knowledge Base Statistics",
        border_style="green"
    ))


def handle_input(user_input: str) -> bool:
    """
    Handle user input.
    Returns True to continue, False to exit.
    """
    user_input = user_input.strip()

    if not user_input:
        return True

    # Check for exit commands
    if user_input.lower() in ('quit', 'exit', 'q'):
        console.print("[dim]Goodbye![/dim]")
        return False

    # Check for stats command
    if user_input.lower() == 'stats':
        print_stats()
        return True

    # Check for clear command with domain (requires confirmation)
    if user_input.lower().startswith('clear '):
        domain_arg = user_input[6:].strip().lower()
        domain_map = {'seo': 'seo', 'web': 'web_builder', 'web_builder': 'web_builder'}
        domain = domain_map.get(domain_arg)

        if not domain:
            console.print(f"[red]Unknown domain: {domain_arg}. Use 'clear seo' or 'clear web'.[/red]")
            return True

        manager = get_embeddings_manager(domain)
        stats = manager.get_stats_by_source()
        domain_name = DOMAINS[domain]['display_name']

        console.print(f"\n[yellow]⚠️  This will delete ALL {stats['total']} chunks from {domain_name}:[/yellow]")
        console.print(f"   • YouTube transcripts: {stats.get('youtube', 0)}")
        console.print(f"   • Local files: {stats.get('file', 0)}")
        console.print(f"   • Local videos: {stats.get('local_video', 0)}")
        console.print(f"\n[bold]Type 'yes' to confirm:[/bold] ", end="")
        confirm = input().strip().lower()
        if confirm == 'yes':
            manager.clear(confirm=True)
        else:
            console.print("[dim]Clear cancelled[/dim]")
        return True

    # Check for transcribe command with domain
    if user_input.lower().startswith('transcribe '):
        rest = user_input[11:].strip()

        # Parse domain
        domain = "seo"  # Default
        if rest.lower().startswith('seo '):
            domain = "seo"
            video_path = rest[4:].strip()
        elif rest.lower().startswith('web '):
            domain = "web_builder"
            video_path = rest[4:].strip()
        else:
            video_path = rest  # Legacy: no domain specified

        video_path = os.path.expanduser(video_path)

        if not os.path.exists(video_path):
            console.print(f"[red]Path not found: {video_path}[/red]")
            return True

        try:
            manager = get_embeddings_manager(domain)
            domain_name = DOMAINS[domain]['display_name']
            total_saved = [0]

            def save_chunks(chunks):
                manager.add_chunks(chunks)
                total_saved[0] += len(chunks)

            if os.path.isdir(video_path):
                console.print(f"[blue]Transcribing videos to {domain_name}: {video_path}[/blue]")
                console.print(f"[dim]Supported formats: {', '.join(VIDEO_EXTENSIONS)}[/dim]")
                console.print("[yellow]Note: This uses local Whisper and may take a while...[/yellow]")
                chunks = ingest_video_folder(video_path, save_callback=save_chunks)
            else:
                console.print(f"[blue]Transcribing to {domain_name}: {video_path}[/blue]")
                chunks = ingest_video(video_path)
                if chunks:
                    save_chunks(chunks)

            if total_saved[0] > 0:
                console.print(f"[green]Successfully transcribed and ingested {total_saved[0]} chunks to {domain_name}![/green]")
            else:
                console.print("[yellow]No content was transcribed.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error transcribing: {e}[/red]")
            import traceback
            traceback.print_exc()
        return True

    # Check for ingest command with domain
    if user_input.lower().startswith('ingest '):
        rest = user_input[7:].strip()

        # Parse domain
        domain = "seo"  # Default
        if rest.lower().startswith('seo '):
            domain = "seo"
            content_path = rest[4:].strip()
        elif rest.lower().startswith('web '):
            domain = "web_builder"
            content_path = rest[4:].strip()
        else:
            content_path = rest  # Legacy: no domain specified

        return ingest_content(content_path, domain)

    # Parse domain prefix for queries (@seo, @web, @all)
    query_domain = None
    query_text = user_input

    if user_input.startswith('@seo '):
        query_domain = "seo"
        query_text = user_input[5:]
    elif user_input.startswith('@web '):
        query_domain = "web_builder"
        query_text = user_input[5:]
    elif user_input.startswith('@all '):
        query_domain = "all"
        query_text = user_input[5:]

    # Check if it's a YouTube URL (for legacy direct pasting)
    if is_youtube_url(query_text) and query_domain is None:
        return ingest_content(query_text, "seo")

    # Check if it's a local path (for legacy direct pasting)
    if is_local_path(query_text) and query_domain is None:
        path = os.path.expanduser(query_text)
        if os.path.exists(path):
            return ingest_content(query_text, "seo")

    # Otherwise, treat it as a query
    try:
        response = query(query_text, domain=query_domain)
        console.print()
        console.print(Markdown(response))
        console.print()
    except Exception as e:
        console.print(f"[red]Error querying: {e}[/red]")

    return True


def ingest_content(content_path: str, domain: str) -> bool:
    """Ingest content (URL or local path) to a specific domain."""
    content_path = content_path.strip()
    domain_name = DOMAINS[domain]['display_name']

    # Check if it's a YouTube URL
    if is_youtube_url(content_path):
        try:
            manager = get_embeddings_manager(domain)
            total_saved = [0]

            def save_chunks(chunks):
                manager.add_chunks(chunks)
                total_saved[0] += len(chunks)

            console.print(f"[blue]Ingesting to {domain_name}: {content_path}[/blue]")
            chunks = ingest_url(content_path, save_callback=save_chunks)
            if total_saved[0] > 0:
                console.print(f"[green]Successfully ingested {total_saved[0]} chunks to {domain_name}![/green]")
            else:
                console.print("[yellow]No content was ingested (no transcripts available).[/yellow]")
        except Exception as e:
            console.print(f"[red]Error ingesting URL: {e}[/red]")
        return True

    # Check if it's a web URL (non-YouTube)
    if is_web_url(content_path):
        try:
            manager = get_embeddings_manager(domain)
            console.print(f"[blue]Ingesting web page to {domain_name}: {content_path}[/blue]")
            chunks = ingest_web_page(content_path)
            if chunks:
                manager.add_chunks(chunks)
                console.print(f"[green]Successfully ingested {len(chunks)} chunks to {domain_name}![/green]")
            else:
                console.print("[yellow]No content was extracted from web page.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error ingesting web page: {e}[/red]")
            import traceback
            traceback.print_exc()
        return True

    # Check if it's a local path
    if is_local_path(content_path):
        path = os.path.expanduser(content_path)
        if os.path.exists(path):
            try:
                manager = get_embeddings_manager(domain)
                total_saved = [0]

                def save_chunks(chunks):
                    manager.add_chunks(chunks)
                    total_saved[0] += len(chunks)

                if os.path.isdir(path):
                    console.print(f"[blue]Ingesting folder to {domain_name}: {path}[/blue]")
                    console.print(f"[dim]Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}[/dim]")
                    chunks = ingest_folder(path, save_callback=save_chunks)
                else:
                    console.print(f"[blue]Ingesting file to {domain_name}: {path}[/blue]")
                    chunks = ingest_file(path)
                    if chunks:
                        save_chunks(chunks)

                if total_saved[0] > 0:
                    console.print(f"[green]Successfully ingested {total_saved[0]} chunks to {domain_name}![/green]")
                else:
                    console.print("[yellow]No content was ingested.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error ingesting path: {e}[/red]")
                import traceback
                traceback.print_exc()
        else:
            console.print(f"[red]Path not found: {path}[/red]")
        return True

    console.print(f"[red]Invalid content path: {content_path}[/red]")
    return True


def main():
    """Main entry point for the CLI."""
    print_welcome()

    while True:
        try:
            user_input = console.input("[bold green]>[/bold green] ")
            if not handle_input(user_input):
                break
        except KeyboardInterrupt:
            console.print("\n[dim]Use 'quit' to exit[/dim]")
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break


if __name__ == "__main__":
    main()
