#!/usr/bin/env python3
"""Script to add 'TestedSEO' label to videos from the TestedSEO folder."""

import sys
sys.path.insert(0, '.')

from src.embeddings import get_embeddings_manager
from rich.console import Console

console = Console()


def main():
    manager = get_embeddings_manager()

    # Get all chunks with metadata
    results = manager.collection.get(include=['metadatas', 'documents', 'embeddings'])

    console.print(f"[blue]Total chunks in database: {len(results['ids'])}[/blue]")

    # Find chunks from TestedSEO folder
    chunks_to_update = []
    for i, (doc_id, meta, doc, embedding) in enumerate(zip(
        results['ids'],
        results['metadatas'],
        results['documents'],
        results['embeddings']
    )):
        if meta and meta.get('source_type') == 'local_video':
            file_path = meta.get('file_path', '')
            if 'TestedSEO' in file_path:
                # Add channel_name to metadata
                new_meta = meta.copy()
                new_meta['channel_name'] = 'TestedSEO'

                # Determine subcategory based on filename
                file_name = meta.get('file_name', '')
                if file_name.startswith('Mastermind'):
                    new_meta['video_series'] = 'Mastermind'
                elif file_name.startswith('Indexation Audit'):
                    new_meta['video_series'] = 'Indexation Audit'
                elif file_name.startswith('Screen Recording'):
                    # Extract week from path
                    if 'Week 4' in file_path:
                        new_meta['video_series'] = 'Week 4'
                    elif 'Week 5' in file_path:
                        new_meta['video_series'] = 'Week 5'
                    elif 'Week 6' in file_path:
                        new_meta['video_series'] = 'Week 6'
                    elif 'Week 8' in file_path:
                        new_meta['video_series'] = 'Week 8'
                    elif 'Week 9' in file_path:
                        new_meta['video_series'] = 'Week 9'
                    elif 'Mastermind' in file_path:
                        new_meta['video_series'] = 'Mastermind'
                    else:
                        new_meta['video_series'] = 'Training'
                else:
                    new_meta['video_series'] = 'Training'

                chunks_to_update.append({
                    'id': doc_id,
                    'metadata': new_meta,
                    'document': doc,
                    'embedding': embedding,
                })

    console.print(f"[blue]TestedSEO chunks to update: {len(chunks_to_update)}[/blue]")

    if not chunks_to_update:
        console.print("[yellow]No TestedSEO chunks found to update[/yellow]")
        return

    # Update in batches
    batch_size = 500
    for i in range(0, len(chunks_to_update), batch_size):
        batch = chunks_to_update[i:i+batch_size]

        ids = [c['id'] for c in batch]
        metadatas = [c['metadata'] for c in batch]
        documents = [c['document'] for c in batch]
        embeddings = [c['embedding'] for c in batch]

        # Delete old entries
        manager.collection.delete(ids=ids)

        # Add back with updated metadata
        manager.collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
            embeddings=embeddings,
        )

        console.print(f"[green]Updated batch {i//batch_size + 1}: {len(batch)} chunks[/green]")

    console.print(f"\n[bold green]Done! Updated {len(chunks_to_update)} TestedSEO chunks[/bold green]")

    # Show sample
    console.print("\n[bold]Sample updated metadata:[/bold]")
    sample = chunks_to_update[:3]
    for s in sample:
        console.print(f"  - {s['metadata'].get('file_name')}")
        console.print(f"    channel_name: {s['metadata'].get('channel_name')}")
        console.print(f"    video_series: {s['metadata'].get('video_series')}")


if __name__ == "__main__":
    main()
