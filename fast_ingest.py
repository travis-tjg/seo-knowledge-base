#!/usr/bin/env python3
"""Fast parallel ingestion with logging to diagnose crashes."""

import json
import sys
import logging
import traceback
import signal
import atexit
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '.')

# Set up logging FIRST
LOG_DIR = Path.home() / ".seo-knowledge-base" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log uncaught exceptions
def exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_handler

# Log signals
def signal_handler(signum, frame):
    logger.error(f"Received signal {signum}")
    sys.exit(1)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Log on exit
def exit_handler():
    logger.info("Script exiting")

atexit.register(exit_handler)

logger.info(f"Logging to {LOG_FILE}")

from config import TRANSCRIPTS_DIR
from src.ingest import (
    TranscriptChunk, chunk_transcript, VideoInfo,
    get_videos_from_channel, process_video_with_proxy, PROXIES, _db_lock
)
from src.embeddings import get_embeddings_manager

CHANNELS = [
    "https://www.youtube.com/@SEOFightClub/streams",
    "https://www.youtube.com/@Digitaleer",
    "https://www.youtube.com/@GBPRoast",
]

def fast_reingest_transcripts():
    """Re-ingest saved transcripts with incremental saves."""
    logger.info("Starting transcript re-ingestion")

    try:
        manager = get_embeddings_manager()
        logger.info("Got embeddings manager")
    except Exception as e:
        logger.error(f"Failed to get embeddings manager: {e}")
        logger.error(traceback.format_exc())
        raise

    transcript_files = list(TRANSCRIPTS_DIR.glob("*.json"))
    youtube_transcripts = [f for f in transcript_files if not f.stem.endswith('_whisper')]
    logger.info(f"Found {len(youtube_transcripts)} saved transcripts")

    total_chunks = 0
    batch = []
    batch_size = 100

    for i, transcript_file in enumerate(youtube_transcripts):
        try:
            video_id = transcript_file.stem
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)

            video_info = VideoInfo(
                video_id=video_id,
                title=f"Video {video_id}",
                channel_name="Unknown",
                video_url=f"https://www.youtube.com/watch?v={video_id}"
            )
            chunks = chunk_transcript(transcript_data, video_info)
            batch.extend(chunks)

        except Exception as e:
            logger.warning(f"Error processing {transcript_file}: {e}")

        # Save batch incrementally
        if (i + 1) % batch_size == 0 and batch:
            try:
                logger.info(f"Saving batch at file {i+1}/{len(youtube_transcripts)}, {len(batch)} chunks")
                manager.add_chunks(batch)
                total_chunks += len(batch)
                logger.info(f"Saved successfully. Total: {total_chunks} chunks")
                batch = []
            except Exception as e:
                logger.error(f"Failed to save batch: {e}")
                logger.error(traceback.format_exc())
                raise

    # Save remaining
    if batch:
        try:
            logger.info(f"Saving final batch: {len(batch)} chunks")
            manager.add_chunks(batch)
            total_chunks += len(batch)
        except Exception as e:
            logger.error(f"Failed to save final batch: {e}")
            logger.error(traceback.format_exc())
            raise

    logger.info(f"Transcript re-ingestion complete: {total_chunks} chunks")
    return total_chunks

def fast_fetch_new_videos():
    """Fetch remaining videos from all channels."""
    logger.info("Starting new video fetch")

    try:
        manager = get_embeddings_manager()
    except Exception as e:
        logger.error(f"Failed to get embeddings manager: {e}")
        logger.error(traceback.format_exc())
        raise

    # Get existing video IDs
    existing_ids = {f.stem for f in TRANSCRIPTS_DIR.glob("*.json") if not f.stem.endswith('_whisper')}
    logger.info(f"Already have {len(existing_ids)} transcripts")

    # Collect ALL new videos from ALL channels
    all_new_videos = []
    for channel_url in CHANNELS:
        logger.info(f"Fetching video list: {channel_url}")
        try:
            videos = get_videos_from_channel(channel_url)
            new_videos = [v for v in videos if v.video_id not in existing_ids]
            logger.info(f"  {len(videos)} total, {len(new_videos)} new")
            all_new_videos.extend(new_videos)
        except Exception as e:
            logger.error(f"Failed to fetch channel {channel_url}: {e}")
            logger.error(traceback.format_exc())

    if not all_new_videos:
        logger.info("No new videos to fetch")
        return 0

    logger.info(f"Fetching {len(all_new_videos)} new videos with 8 parallel workers")

    total_chunks = [0]
    processed = [0]
    errors = [0]

    def save_chunks(chunks):
        try:
            manager.add_chunks(chunks)
            total_chunks[0] += len(chunks)
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            errors[0] += 1

    # Process videos in parallel
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for i, video in enumerate(all_new_videos):
                proxy = PROXIES[i % len(PROXIES)]
                future = executor.submit(process_video_with_proxy, video, proxy, save_chunks)
                futures[future] = video

            for future in as_completed(futures):
                processed[0] += 1
                video = futures[future]
                try:
                    result = future.result()
                    if processed[0] % 10 == 0:
                        logger.info(f"Progress: {processed[0]}/{len(all_new_videos)} videos, {total_chunks[0]} chunks, {errors[0]} errors")
                except Exception as e:
                    logger.error(f"Error processing {video.title}: {e}")
                    errors[0] += 1

    except Exception as e:
        logger.error(f"ThreadPoolExecutor error: {e}")
        logger.error(traceback.format_exc())
        raise

    logger.info(f"Video fetch complete: {total_chunks[0]} chunks, {errors[0]} errors")
    return total_chunks[0]

def main():
    logger.info("="*60)
    logger.info("FAST INGEST - Starting")
    logger.info("="*60)

    try:
        # Step 1: Re-ingest saved transcripts
        chunks_from_saved = fast_reingest_transcripts()

        # Step 2: Fetch new videos
        chunks_from_new = fast_fetch_new_videos()

        # Final stats
        manager = get_embeddings_manager()
        stats = manager.get_stats_by_source()
        logger.info("="*60)
        logger.info("COMPLETE!")
        logger.info(f"  From saved transcripts: {chunks_from_saved}")
        logger.info(f"  From new videos: {chunks_from_new}")
        logger.info(f"  Total in DB: {stats['total']}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Main function error: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
