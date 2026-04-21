"""
Central Gemini API client — multi-key, multi-model with automatic failover.

Configure via .env:
    GEMINI_API_KEY=<primary key>
    GEMINI_API_KEY_2=<second key>        # optional
    GEMINI_API_KEY_3=<third key>         # optional
    GEMINI_MODEL=gemini-1.5-flash        # primary model (switch anytime)
    GEMINI_API_ENDPOINT=<optional full override URL>

Failover order: try each model across ALL keys before moving to the next model.
Example with 3 keys: key1/primary → key2/primary → key3/primary → key1/fallback1 → ...
This finds a working key in 1-3 calls instead of burning through 12 models on key1 first.
Free tier: ~1500 requests/day per key. Quota resets daily at midnight Pacific.
"""

import os
import requests
import time
import json
import re

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-pro",
]

# Module-level cache — 404 models are dead for the entire server session, no need to retry them
_dead_models: set[str] = set()

def _get_api_keys() -> list[str]:
    """Collect all configured API keys in priority order."""
    keys = []
    # Primary key
    primary = os.environ.get("GEMINI_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    # Additional keys: GEMINI_API_KEY_2, GEMINI_API_KEY_3, ... up to 10
    for i in range(2, 11):
        k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    return keys

def safe_json_parse(text: str) -> dict | None:
    if not text:
        return None
    try:
        clean_text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except:
        pass
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return None

def get_gemini_url(model: str, api_key: str) -> str:
    endpoint = os.environ.get("GEMINI_API_ENDPOINT")
    if endpoint:
        return f"{endpoint}?key={api_key}"
    return f"{BASE_URL}/{model}:generateContent?key={api_key}"

def call_gemini(payload: dict, timeout: int = 30, caller: str = "") -> dict | None:
    """
    Multi-key, multi-model failover.
    Loop order: model-outer, key-inner — so primary model is tried on all keys
    before falling back to the next model. Minimises wasted calls when one key
    has quota and others are exhausted.
    """
    api_keys = _get_api_keys()
    if not api_keys:
        print(f"{caller} [Gemini] ERROR: No API key set. Add GEMINI_API_KEY to .env")
        return None

    primary_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    for model_index, model in enumerate(models_to_try):
        if model in _dead_models:
            continue

        for key_index, api_key in enumerate(api_keys):
            key_label = f"key{key_index + 1}"
            url = get_gemini_url(model, api_key)

            for attempt in range(2):
                try:
                    resp = requests.post(
                        url, json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=timeout
                    )

                    if resp.status_code == 200:
                        if model_index > 0 or key_index > 0:
                            print(f"{caller} [Gemini] Success — {key_label}, model: {model}")
                        return resp.json()

                    if resp.status_code in [429, 503]:
                        print(f"{caller} [Gemini] {key_label} / '{model}' quota/busy (HTTP {resp.status_code}). Trying next key...")
                        break  # try same model on next key

                    if resp.status_code == 404:
                        _dead_models.add(model)
                        print(f"{caller} [Gemini] '{model}' not found — skipping for all keys (cached).")
                        break  # skip all keys for this model

                    if resp.status_code == 500 and attempt == 0:
                        time.sleep(1)
                        continue

                    print(f"{caller} [Gemini] HTTP {resp.status_code}: {resp.text[:300]}")
                    break

                except requests.exceptions.Timeout:
                    print(f"{caller} [Gemini] {key_label} / '{model}' timed out. Trying next key...")
                    break
                except Exception as e:
                    print(f"{caller} [Gemini] {key_label} / '{model}' connection error: {e}")
                    time.sleep(1)
                    continue

            if model in _dead_models:
                break  # stop trying other keys for a 404 model

    total_keys = len(api_keys)
    print(f"{caller} [Gemini] All {total_keys} key(s) and all models exhausted. Add more keys or wait for quota reset (midnight PT).")
    return None
