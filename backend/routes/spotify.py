"""Music & Story Search API Routes.

Searches YouTube for songs, mood-based content, and story videos.
Primary: YouTube Data API v3 (reliable, quota-based).
Fallback: HTML scrape (used automatically if API fails or quota is exceeded).
"""

from fastapi import APIRouter, Query
import urllib.request
import urllib.parse
import urllib.error
import re
import os
import json

from services.voice_bot_engine import get_content_recommendation, get_story_queries, STORY_CATEGORIES

router = APIRouter(tags=["Music Search"])


# ─────────────────────────────────────────────
# YouTube Data API v3 — primary method
# ─────────────────────────────────────────────
def _youtube_api_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search using the official YouTube Data API v3.
    Returns a list of result dicts, or empty list on failure/quota exceeded.
    """
    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not api_key:
        return []

    try:
        params = urllib.parse.urlencode({
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "key": api_key,
            "relevanceLanguage": "hi",
            "regionCode": "IN"
        })
        url = "https://www.googleapis.com/youtube/v3/search?" + params
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=10).read().decode()
        data = json.loads(raw)

        results = []
        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue
            title = item.get("snippet", {}).get("title", query.title())
            channel = item.get("snippet", {}).get("channelTitle", "YouTube")
            results.append({
                "name": title,
                "artist": channel,
                "youtube_id": video_id,
                "url": f"https://www.youtube.com/embed/{video_id}?autoplay=1"
            })

        if results:
            print(f"[YouTube API] Found {len(results)} results for: {query!r}")
        return results

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 403 and ("quotaExceeded" in body or "keyInvalid" in body):
            print(f"[YouTube API] Quota/key error for {query!r} — falling back to scrape")
        else:
            print(f"[YouTube API] HTTP {e.code} for {query!r}: {body[:200]}")
        return []
    except Exception as e:
        print(f"[YouTube API] Error for {query!r}: {e}")
        return []


# ─────────────────────────────────────────────
# HTML scrape fallback
# ─────────────────────────────────────────────
def _youtube_scrape(query: str, suffix: str = "") -> dict | None:
    """
    Fallback: scrape YouTube search page for first video ID.
    Used automatically when the API fails or quota is exceeded.
    """
    try:
        q = urllib.parse.urlencode({"search_query": f"{query} {suffix}".strip()})
        req = urllib.request.Request(
            "https://www.youtube.com/results?" + q,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        html = urllib.request.urlopen(req, timeout=10).read().decode()
        ids = re.findall(r'watch\?v=([a-zA-Z0-9_-]{11})', html)
        if ids:
            print(f"[YouTube Scrape] Found result for: {query!r}")
            return {
                "name": query.title(),
                "artist": "YouTube",
                "youtube_id": ids[0],
                "url": f"https://www.youtube.com/embed/{ids[0]}?autoplay=1"
            }
    except Exception as e:
        print(f"[YouTube Scrape] Error for {query!r}: {e}")
    return None


# ─────────────────────────────────────────────
# Unified search: API → scrape fallback
# ─────────────────────────────────────────────
def _youtube_search(query: str, suffix: str = "") -> dict | None:
    """
    Try YouTube Data API v3 first; fall back to HTML scrape automatically.
    Always returns a single best result dict, or None.
    """
    full_query = f"{query} {suffix}".strip()

    # 1. Try official API
    api_results = _youtube_api_search(full_query, max_results=1)
    if api_results:
        return api_results[0]

    # 2. Fallback to HTML scrape
    return _youtube_scrape(query, suffix=suffix)


def _youtube_search_many(query: str, max_results: int = 3) -> list[dict]:
    """
    Return up to max_results results. API first, scrape as fallback (1 result).
    """
    full_query = query.strip()

    api_results = _youtube_api_search(full_query, max_results=max_results)
    if api_results:
        return api_results

    # Scrape fallback gives only 1 result
    single = _youtube_scrape(full_query)
    return [single] if single else []


# ─────────────────────────────────────────────
# Route: generic search
# ─────────────────────────────────────────────
@router.get("/music/youtube")
def search_youtube_track(q: str = Query(..., min_length=1, description="Search query")):
    """Search YouTube and return the best matching video for a song or story."""
    results = _youtube_search_many(q, max_results=3)
    source = "youtube_api" if results and results[0].get("artist") != "YouTube" else "youtube_scrape"
    if results:
        return {"tracks": results, "source": source}
    return {"tracks": [], "source": source, "error": "No results found"}


# ─────────────────────────────────────────────
# Route: mood-based music
# ─────────────────────────────────────────────
@router.get("/music/mood")
def search_by_mood(mood: str = Query(..., description="User mood")):
    """Return YouTube video suggestions based on user mood."""
    rec = get_content_recommendation(mood)
    tracks = []
    for query in rec.get("queries", []):
        result = _youtube_search(query)
        if result:
            result["name"] = query.title()
            tracks.append(result)
    return {
        "tracks": tracks,
        "mood": mood,
        "content_type": rec.get("type", "music"),
        "message": rec.get("message", ""),
        "source": "youtube"
    }


# ─────────────────────────────────────────────
# Route: story search
# ─────────────────────────────────────────────
@router.get("/story/youtube")
def search_story(category: str = Query("moral", description="Story category")):
    """Search YouTube for a story/katha video by category."""
    queries = get_story_queries(category)
    tracks = []
    for query in queries:
        result = _youtube_search(query)
        if result:
            result["name"] = query.title()
            result["category"] = category
            tracks.append(result)
    return {
        "tracks": tracks,
        "category": category,
        "available_categories": list(STORY_CATEGORIES.keys()),
        "source": "youtube"
    }


# ─────────────────────────────────────────────
# Route: list categories
# ─────────────────────────────────────────────
@router.get("/story/categories")
def list_story_categories():
    """List all available story categories."""
    return {"categories": list(STORY_CATEGORIES.keys())}


# Legacy alias
@router.get("/spotify/search")
def search_tracks(q: str = Query(...)):
    return search_youtube_track(q)
