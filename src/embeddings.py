"""Embedding generation and ChromaDB operations."""

from datetime import datetime
from typing import Union
import chromadb
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rich.console import Console

import sys
sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, OPENAI_API_KEY, DOMAINS, DEFAULT_DOMAIN
from src.ingest import TranscriptChunk
from src.file_ingest import FileChunk
from src.video_ingest import VideoChunk
from src.web_ingest import WebChunk

console = Console()


class EmbeddingsManager:
    """Manages embeddings and ChromaDB vector store for a specific domain."""

    def __init__(self, domain: str = None):
        """
        Initialize the embeddings manager.

        Args:
            domain: The domain to use (e.g., 'seo', 'web_builder').
                    If None, uses DEFAULT_DOMAIN for backward compatibility.
        """
        # Determine domain and collection name
        if domain is None:
            # Backward compatibility: use legacy COLLECTION_NAME
            self.domain = DEFAULT_DOMAIN
            self.collection_name = COLLECTION_NAME
        else:
            self.domain = domain
            domain_config = DOMAINS.get(domain, DOMAINS[DEFAULT_DOMAIN])
            self.collection_name = domain_config["collection_name"]

        # Configure LlamaIndex settings
        Settings.embed_model = OpenAIEmbedding(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # Load or create index
        self._index = None

    @property
    def index(self) -> VectorStoreIndex:
        """Get or create the vector store index."""
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context
            )
        return self._index

    def add_chunks(self, chunks: list[Union[TranscriptChunk, FileChunk, VideoChunk, WebChunk]], boost: float = 1.0) -> int:
        """
        Add transcript, file, or video chunks to the vector store.

        Args:
            chunks: List of chunks to add
            boost: Relevance boost multiplier (e.g., 1.15 for 15% boost).
                   This is stored in metadata and applied during retrieval.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        # Common metadata for all chunks in this batch
        ingested_at = datetime.now().isoformat()

        nodes = []
        for chunk in chunks:
            if isinstance(chunk, TranscriptChunk):
                node = TextNode(
                    text=chunk.text,
                    metadata={
                        'source_type': 'youtube',
                        'video_id': chunk.video_id,
                        'video_title': chunk.video_title,
                        'channel_name': chunk.channel_name,
                        'video_url': chunk.video_url,
                        'start_time': chunk.start_time,
                        'end_time': chunk.end_time,
                        'timestamp_url': chunk.timestamp_url,
                        'boost': boost,
                        'ingested_at': ingested_at,
                    },
                    excluded_embed_metadata_keys=[
                        'video_url',
                        'timestamp_url',
                        'video_id',
                        'source_type',
                        'boost',
                        'ingested_at',
                    ],
                    excluded_llm_metadata_keys=[
                        'video_id',
                        'source_type',
                        'boost',
                        'ingested_at',
                    ],
                )
            elif isinstance(chunk, FileChunk):
                node = TextNode(
                    text=chunk.text,
                    metadata={
                        'source_type': 'file',
                        'file_path': chunk.file_path,
                        'file_name': chunk.file_name,
                        'page_number': chunk.page_number,
                        'chunk_index': chunk.chunk_index,
                        'boost': boost,
                        'ingested_at': ingested_at,
                    },
                    excluded_embed_metadata_keys=[
                        'file_path',
                        'source_type',
                        'chunk_index',
                        'boost',
                        'ingested_at',
                    ],
                    excluded_llm_metadata_keys=[
                        'source_type',
                        'chunk_index',
                        'boost',
                        'ingested_at',
                    ],
                )
            elif isinstance(chunk, VideoChunk):
                # Format timestamp for display (MM:SS or HH:MM:SS)
                start_mins = int(chunk.start_time // 60)
                start_secs = int(chunk.start_time % 60)
                if start_mins >= 60:
                    hours = start_mins // 60
                    mins = start_mins % 60
                    timestamp = f"{hours}:{mins:02d}:{start_secs:02d}"
                else:
                    timestamp = f"{start_mins}:{start_secs:02d}"

                node = TextNode(
                    text=chunk.text,
                    metadata={
                        'source_type': 'local_video',
                        'file_path': chunk.file_path,
                        'file_name': chunk.file_name,
                        'start_time': chunk.start_time,
                        'end_time': chunk.end_time,
                        'timestamp': timestamp,
                        'chunk_index': chunk.chunk_index,
                        'boost': boost,
                        'ingested_at': ingested_at,
                    },
                    excluded_embed_metadata_keys=[
                        'file_path',
                        'source_type',
                        'chunk_index',
                        'end_time',
                        'boost',
                        'ingested_at',
                    ],
                    excluded_llm_metadata_keys=[
                        'source_type',
                        'chunk_index',
                        'boost',
                        'ingested_at',
                    ],
                )
            elif isinstance(chunk, WebChunk):
                node = TextNode(
                    text=chunk.text,
                    metadata={
                        'source_type': 'web',
                        'url': chunk.url,
                        'video_title': chunk.title,  # Reuse video_title for consistency
                        'channel_name': chunk.domain or 'Web',  # Reuse channel_name for domain
                        'timestamp_url': chunk.url,  # Link back to page
                        'start_time': 0,  # No timestamp for web
                        'chunk_index': chunk.chunk_index,
                        'boost': boost,
                        'ingested_at': ingested_at,
                    },
                    excluded_embed_metadata_keys=[
                        'url',
                        'source_type',
                        'chunk_index',
                        'timestamp_url',
                        'boost',
                        'ingested_at',
                    ],
                    excluded_llm_metadata_keys=[
                        'source_type',
                        'chunk_index',
                        'boost',
                        'ingested_at',
                    ],
                )
            else:
                continue
            nodes.append(node)

        console.print(f"[blue]Generating embeddings for {len(nodes)} chunks...[/blue]")

        # Create a new index with these nodes (adds to existing vector store)
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            show_progress=True
        )

        console.print(f"[green]Added {len(nodes)} chunks to vector store[/green]")
        return len(nodes)

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'domain': self.domain,
            'embedding_model': EMBEDDING_MODEL,
        }

    def get_stats_by_source(self) -> dict:
        """Get detailed statistics broken down by source type."""
        results = self.collection.get(include=['metadatas'])

        stats = {
            'youtube': 0,
            'file': 0,
            'local_video': 0,
            'web': 0,
            'unknown': 0,
        }

        for metadata in results['metadatas']:
            if metadata:
                source_type = metadata.get('source_type', 'unknown')
                stats[source_type] = stats.get(source_type, 0) + 1
            else:
                stats['unknown'] += 1

        stats['total'] = sum(stats.values())
        return stats

    def preview_delete(self, source_type: str = None, filename_patterns: list[str] = None) -> dict:
        """
        Preview what would be deleted without actually deleting.
        Returns counts of what would be affected.
        """
        results = self.collection.get(include=['metadatas'])

        would_delete = []
        would_keep = []

        for doc_id, metadata in zip(results['ids'], results['metadatas']):
            should_delete = False

            if source_type and metadata:
                if metadata.get('source_type') == source_type:
                    should_delete = True

            if filename_patterns and metadata:
                file_name = metadata.get('file_name', '')
                if any(pattern in file_name for pattern in filename_patterns):
                    should_delete = True

            if should_delete:
                would_delete.append(doc_id)
            else:
                would_keep.append(doc_id)

        return {
            'would_delete': len(would_delete),
            'would_keep': len(would_keep),
            'total': len(results['ids']),
        }

    def delete_by_source_type(self, source_type: str, confirm: bool = False) -> int:
        """
        Delete all chunks of a specific source type.
        Requires confirm=True to actually delete.
        Returns the number of chunks deleted (or would be deleted if confirm=False).
        """
        results = self.collection.get(include=['metadatas'])

        ids_to_delete = []
        for doc_id, metadata in zip(results['ids'], results['metadatas']):
            if metadata and metadata.get('source_type') == source_type:
                ids_to_delete.append(doc_id)

        if not confirm:
            console.print(f"[yellow]DRY RUN: Would delete {len(ids_to_delete)} chunks with source_type='{source_type}'[/yellow]")
            console.print(f"[dim]Pass confirm=True to actually delete[/dim]")
            return len(ids_to_delete)

        if ids_to_delete:
            # Delete in batches
            batch_size = 5000
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i+batch_size]
                self.collection.delete(ids=batch)
            self._index = None
            console.print(f"[yellow]Deleted {len(ids_to_delete)} chunks with source_type='{source_type}'[/yellow]")
        else:
            console.print(f"[dim]No chunks found with source_type='{source_type}'[/dim]")

        return len(ids_to_delete)

    def delete_by_filename_pattern(self, patterns: list[str], confirm: bool = False) -> int:
        """
        Delete chunks where file_name contains any of the given patterns.
        Requires confirm=True to actually delete.
        Returns the number of chunks deleted (or would be deleted if confirm=False).
        """
        results = self.collection.get(include=['metadatas'])

        ids_to_delete = []
        for doc_id, metadata in zip(results['ids'], results['metadatas']):
            if metadata and 'file_name' in metadata:
                file_name = metadata['file_name']
                if any(pattern in file_name for pattern in patterns):
                    ids_to_delete.append(doc_id)

        if not confirm:
            console.print(f"[yellow]DRY RUN: Would delete {len(ids_to_delete)} chunks matching patterns: {patterns}[/yellow]")
            console.print(f"[dim]Pass confirm=True to actually delete[/dim]")
            return len(ids_to_delete)

        if ids_to_delete:
            # Delete in batches
            batch_size = 5000
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i+batch_size]
                self.collection.delete(ids=batch)
            self._index = None
            console.print(f"[yellow]Deleted {len(ids_to_delete)} chunks matching patterns: {patterns}[/yellow]")
        else:
            console.print(f"[dim]No chunks found matching patterns: {patterns}[/dim]")

        return len(ids_to_delete)

    def clear(self, confirm: bool = False) -> None:
        """
        Clear all data from the vector store.
        Requires confirm=True to actually clear.
        """
        if not confirm:
            count = self.collection.count()
            console.print(f"[yellow]DRY RUN: Would delete ALL {count} chunks from the {self.domain} database[/yellow]")
            console.print(f"[dim]Pass confirm=True to actually clear[/dim]")
            return

        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self._index = None
        console.print(f"[yellow]Vector store for {self.domain} cleared[/yellow]")


# Multi-domain singleton instances
_managers: dict[str, EmbeddingsManager] = {}

# Legacy singleton for backward compatibility
_manager: EmbeddingsManager | None = None


def get_embeddings_manager(domain: str = None) -> EmbeddingsManager:
    """
    Get the embeddings manager for a specific domain.

    Args:
        domain: The domain to use (e.g., 'seo', 'web_builder').
                If None, returns the legacy singleton for backward compatibility.

    Returns:
        EmbeddingsManager instance for the specified domain.
    """
    global _managers, _manager

    if domain is None:
        # Backward compatibility: return legacy singleton
        if _manager is None:
            _manager = EmbeddingsManager()
        return _manager

    # Multi-domain support
    if domain not in _managers:
        _managers[domain] = EmbeddingsManager(domain)
    return _managers[domain]
