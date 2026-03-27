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

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def get_gemini_url() -> str:
    """Build the Gemini generateContent URL from env vars.
    GEMINI_API_ENDPOINT overrides everything (legacy support).
    Otherwise uses GEMINI_MODEL (defaults to gemini-2.5-flash-lite).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    # Full endpoint override (legacy)
    endpoint = os.environ.get("GEMINI_API_ENDPOINT")
    if endpoint:
        return f"{endpoint}?key={api_key}"
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    return f"{BASE_URL}/{model}:generateContent?key={api_key}"


def call_gemini(payload: dict, timeout: int = 30, caller: str = "") -> dict | None:
    """Make a Gemini API call and return the parsed JSON, or None on failure.

    Logs quota errors clearly so you know when to switch models.
    Args:
        payload: The full request body dict (contents, generationConfig, etc.)
        timeout: Request timeout in seconds
        caller: Label for log output, e.g. '[voice_bot]'
    Returns:
        Parsed JSON dict on success, None on failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(f"{caller} [Gemini] ERROR: GEMINI_API_KEY is not set in .env")
        return None

    url = get_gemini_url()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429:
            try:
                err = resp.json().get("error", {})
                msg = err.get("message", "")
            except Exception:
                msg = resp.text
            print(
                f"\n{'='*60}\n"
                f"{caller} [Gemini] QUOTA EXCEEDED for model '{model}'\n"
                f"  To fix: change GEMINI_MODEL in .env to another model and restart.\n"
                f"  Available free-tier models: gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.0-flash\n"
                f"  Detail: {msg[:300]}\n"
                f"{'='*60}\n"
            )
            return None

        print(f"{caller} [Gemini] HTTP {resp.status_code}: {resp.text[:300]}")
        return None

    except requests.exceptions.Timeout:
        print(f"{caller} [Gemini] Request timed out after {timeout}s (model: {model})")
        return None
    except Exception as e:
        print(f"{caller} [Gemini] Unexpected error: {e}")
        return None
