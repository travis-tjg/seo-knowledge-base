#!/bin/bash
# Cron wrapper script for SEO Fight Club weekly update
# Runs every Tuesday at 4 PM to ingest the latest show

# Change to the project directory
cd "/Users/travisgodec/Library/CloudStorage/GoogleDrive-travisgodec@gmail.com/My Drive/Macbook Pro/Travis' Clone/seo-knowledge-base"

# Activate virtual environment
source venv/bin/activate

# Run the update script (latest video only for weekly update)
python update_seo_fight_club.py --latest >> logs/seo_fight_club_update.log 2>&1

# Deactivate
deactivate
