#!/usr/bin/env python3
"""Test OpenAI API."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print(f"API Key starts with: {os.getenv('OPENAI_API_KEY', '')[:20]}...")

import httpx

response = httpx.post(
    "https://api.openai.com/v1/embeddings",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    },
    json={"input": "test", "model": "text-embedding-3-small"},
    timeout=30.0
)
print(f"Status: {response.status_code}")
print(f"Response: {response.text[:200]}")
