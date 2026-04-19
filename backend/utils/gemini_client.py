"""
Central Gemini API client.

Configure via .env:
    GEMINI_API_KEY=<your key>
    GEMINI_MODEL=gemini-2.5-flash-lite   # switch model here anytime
    GEMINI_API_ENDPOINT=<optional full override URL>

Switch model:  just change GEMINI_MODEL in .env and restart the server.
Available models (free tier): gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash
"""

import os
import requests
import time
import json
import re

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Optimized model order: Stable models first, slower preview models last.
FALLBACK_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-flash-latest"
]

def safe_json_parse(text: str) -> dict | None:
    """
    Tries to parse text as JSON. If it fails, uses regex to extract 
    the first JSON block { ... } and parses that.
    """
    if not text: return None
    
    # 1. Direct parse
    try:
        # Clean up common LLM artifacts
        clean_text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except:
        pass
        
    # 2. Regex extraction (for when LLM adds markdown or chat text)
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
        
    return None

def get_gemini_url(model: str, api_key: str) -> str:
    """Build the Gemini URL for a specific model."""
    endpoint = os.environ.get("GEMINI_API_ENDPOINT")
    if endpoint:
        return f"{endpoint}?key={api_key}"
    return f"{BASE_URL}/{model}:generateContent?key={api_key}"

def call_gemini(payload: dict, timeout: int = 30, caller: str = "") -> dict | None:
    """
    Make a Gemini API call with automatic model switching on failure.
    Includes retry logic for transient network issues.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(f"{caller} [Gemini] ERROR: GEMINI_API_KEY is not set in .env")
        return None

    # Get primary model from env
    primary_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    
    # Create the sequence of models to try
    models_to_try = [primary_model]
    for model in FALLBACK_MODELS:
        if model not in models_to_try:
            models_to_try.append(model)

    for i, model in enumerate(models_to_try):
        url = get_gemini_url(model, api_key)
        
        # Internal retries for the SAME model in case of temporary 5xx or connection error
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)

                if resp.status_code == 200:
                    if i > 0:
                        print(f"{caller} [Gemini] Success using fallback model: {model}")
                    return resp.json()

                # Quota errors (429) or Server Busy (503) -> Try next model immediately
                if resp.status_code in [429, 503]:
                    print(f"{caller} [Gemini] Model '{model}' overloaded (HTTP {resp.status_code}). Trying next...")
                    break 

                # Deprecated model (404) -> Try next
                if resp.status_code == 404:
                    print(f"{caller} [Gemini] Model '{model}' not found. Skipping...")
                    break

                # Internal Error (500) -> Wait 1s and retry ONCE before switching
                if resp.status_code == 500 and attempt == 0:
                    time.sleep(1)
                    continue

                print(f"{caller} [Gemini] HTTP {resp.status_code}: {resp.text[:300]}")
                break # Non-recoverable or already retried

            except requests.exceptions.Timeout:
                print(f"{caller} [Gemini] Timeout for model '{model}'. Trying next...")
                break # Switch model on timeout
            except Exception as e:
                print(f"{caller} [Gemini] Connection error with model '{model}': {e}")
                time.sleep(1)
                continue # Try second attempt for the same model

    print(f"{caller} [Gemini] All models failed. Please check your API key and quotas.")
    return None
