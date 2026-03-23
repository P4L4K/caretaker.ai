"""Music Search API Route.

Searches for full songs using YouTube and returns the first Video ID.
This allows the frontend to embed a YouTube iframe for full song playback.
"""

from fastapi import APIRouter, Query
import urllib.request
import urllib.parse
import re

router = APIRouter(tags=["Music Search"])

@router.get("/music/youtube")
def search_youtube_track(q: str = Query(..., min_length=1, description="Search query")):
    """
    Search YouTube and return the best matching video ID.
    Perfect for playing full Bollywood songs.
    """
    try:
        # Append 'audio' to get tracks that usually allow embedding
        query_string = urllib.parse.urlencode({"search_query": q + " audio lyric"})
        req = urllib.request.Request(
            "https://www.youtube.com/results?" + query_string,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        html_content = urllib.request.urlopen(req, timeout=10)
        search_results = re.findall(r'watch\?v=([a-zA-Z0-9_-]{11})', html_content.read().decode())
        
        if search_results:
            video_id = search_results[0]
            # Since we just scraped HTML, we don't have exact title/art easily.
            # But the iframe handles the UI! We'll just return the ID.
            return {
                "tracks": [{
                    "name": q.title(), # Just return the query as title
                    "artist": "YouTube Search",
                    "youtube_id": video_id,
                    "url": f"https://www.youtube.com/embed/{video_id}?autoplay=1"
                }],
                "source": "youtube"
            }
            
        return {"tracks": [], "source": "youtube", "error": "No results found"}

    except Exception as e:
        print(f"[YouTube] Search error: {e}")
        return {"tracks": [], "source": "youtube", "error": str(e)}

# Keep the old spotify endpoint so we don't break anything expecting it right now
@router.get("/spotify/search")
def search_tracks(q: str = Query(...)):
    return search_youtube_track(q)
