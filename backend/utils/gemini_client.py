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

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# List of working models to try if the primary one fails
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.5-pro",
    "gemini-3.1-flash-lite-preview"
]

def get_gemini_url(model: str, api_key: str) -> str:
    """Build the Gemini URL for a specific model."""
    endpoint = os.environ.get("GEMINI_API_ENDPOINT")
    if endpoint:
        return f"{endpoint}?key={api_key}"
    return f"{BASE_URL}/{model}:generateContent?key={api_key}"

def call_gemini(payload: dict, timeout: int = 30, caller: str = "") -> dict | None:
    """Make a Gemini API call with automatic model switching on failure."""
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
        
        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)

            if resp.status_code == 200:
                if i > 0:
                    print(f"{caller} [Gemini] Success using fallback model: {model}")
                return resp.json()

            # Quota, server errors, or Not Found (deprecated models) trigger a switch
            if resp.status_code in [404, 429, 503, 500]:
                print(f"{caller} [Gemini] Model '{model}' failed (HTTP {resp.status_code}). Trying next model...")
                continue
            
            # For 400 (Invalid/Expired Key), switching models won't help, but we show the error
            if resp.status_code == 400:
                print(f"{caller} [Gemini] HTTP 400 (Bad Request/Expired Key): {resp.text[:200]}")
                return None

            print(f"{caller} [Gemini] HTTP {resp.status_code}: {resp.text[:300]}")
            return None

        except requests.exceptions.Timeout:
            print(f"{caller} [Gemini] Timeout for model '{model}'. Trying next...")
            continue
        except Exception as e:
            print(f"{caller} [Gemini] Unexpected error with model '{model}': {e}")
            time.sleep(2)  # Delay between retries
            continue

    print(f"{caller} [Gemini] All models failed. Please check your API key and quotas.")
    return None
