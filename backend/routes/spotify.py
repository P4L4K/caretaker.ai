"""Music & Story Search API Routes.

Searches YouTube for songs, mood-based content, and story videos.
Returns video IDs for frontend iframe embedding.
"""

from fastapi import APIRouter, Query
import urllib.request
import urllib.parse
import re

from services.voice_bot_engine import get_content_recommendation, get_story_queries, STORY_CATEGORIES

router = APIRouter(tags=["Music Search"])


def _youtube_search(query: str, suffix: str = "audio lyric") -> dict | None:
    """Core YouTube scrape. Returns first result or None."""
    try:
        q = urllib.parse.urlencode({"search_query": f"{query} {suffix}".strip()})
        req = urllib.request.Request(
            "https://www.youtube.com/results?" + q,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        html = urllib.request.urlopen(req, timeout=10).read().decode()
        ids = re.findall(r'watch\?v=([a-zA-Z0-9_-]{11})', html)
        if ids:
            return {
                "name": query.title(),
                "artist": "YouTube",
                "youtube_id": ids[0],
                "url": f"https://www.youtube.com/embed/{ids[0]}?autoplay=1"
            }
    except Exception as e:
        print(f"[YouTube] Search error for '{query}': {e}")
    return None


@router.get("/music/youtube")
def search_youtube_track(q: str = Query(..., min_length=1, description="Search query")):
    """Search YouTube and return the best matching video for a song."""
    result = _youtube_search(q, suffix="audio lyric")
    if result:
        return {"tracks": [result], "source": "youtube"}
    return {"tracks": [], "source": "youtube", "error": "No results found"}


@router.get("/music/mood")
def search_by_mood(mood: str = Query(..., description="User mood")):
    """Return YouTube video suggestions based on user mood."""
    rec = get_content_recommendation(mood)
    tracks = []
    for query in rec.get("queries", []):
        result = _youtube_search(query, suffix="")
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


@router.get("/story/youtube")
def search_story(category: str = Query("moral", description="Story category: historical, mythological, comedy, moral, spiritual")):
    """Search YouTube for a story/katha video by category."""
    queries = get_story_queries(category)
    tracks = []
    for query in queries:
        result = _youtube_search(query, suffix="")
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


@router.get("/story/categories")
def list_story_categories():
    """List all available story categories."""
    return {"categories": list(STORY_CATEGORIES.keys())}


# Keep legacy endpoints for backward compatibility
@router.get("/spotify/search")
def search_tracks(q: str = Query(...)):
    return search_youtube_track(q)
