"""
Spotify API tools for development/learning purposes.
This is NOT part of the core framework - delete when done experimenting.
"""

import os
import requests
import base64
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from ..tools import tool_registry

Registry = tool_registry()


class SpotifySearchParams(BaseModel):
    query: str = Field(..., description="Search query (song, artist, album)")
    search_type: str = Field(
        "track", description="Type: track, artist, album, playlist"
    )
    limit: int = Field(5, description="Number of results (1-50)")


class SpotifyTrackInfoParams(BaseModel):
    track_id: str = Field(..., description="Spotify track ID")


def get_spotify_token() -> Optional[str]:
    """Get Spotify access token using client credentials flow."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        return None

    # Encode credentials
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"grant_type": "client_credentials"}

    try:
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            headers=headers,
            data=data,
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print(f"Failed to get Spotify token: {e}")
        return None


@Registry.tool(description="Search Spotify for tracks, artists, albums, or playlists")
def spotify_search_tool(params: SpotifySearchParams, context: Dict) -> Dict:
    """Search Spotify for tracks, artists, albums, or playlists."""

    token = get_spotify_token()
    if not token:
        return {
            "error": "Spotify API not configured. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables."
        }

    headers = {"Authorization": f"Bearer {token}"}

    search_params = {
        "q": params.query,
        "type": params.search_type,
        "limit": min(params.limit, 50),  # Spotify max is 50
    }

    try:
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params=search_params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        # Extract relevant info based on search type
        results = []
        items_key = f"{params.search_type}s"  # tracks, artists, albums, playlists

        if items_key in data and "items" in data[items_key]:
            for item in data[items_key]["items"]:
                if params.search_type == "track":
                    results.append(
                        {
                            "name": item["name"],
                            "artist": ", ".join(
                                [artist["name"] for artist in item["artists"]]
                            ),
                            "album": item["album"]["name"],
                            "id": item["id"],
                            "url": item["external_urls"]["spotify"],
                            "popularity": item.get("popularity", 0),
                        }
                    )
                elif params.search_type == "artist":
                    results.append(
                        {
                            "name": item["name"],
                            "id": item["id"],
                            "url": item["external_urls"]["spotify"],
                            "followers": item["followers"]["total"],
                            "popularity": item.get("popularity", 0),
                        }
                    )
                elif params.search_type == "album":
                    results.append(
                        {
                            "name": item["name"],
                            "artist": ", ".join(
                                [artist["name"] for artist in item["artists"]]
                            ),
                            "id": item["id"],
                            "url": item["external_urls"]["spotify"],
                            "release_date": item.get("release_date", "Unknown"),
                        }
                    )

        return {
            "results": results,
            "total_found": data[items_key]["total"],
            "query": params.query,
            "search_type": params.search_type,
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Spotify API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@Registry.tool(
    description="Get detailed information about a specific Spotify track by ID"
)
def spotify_track_info_tool(params: SpotifyTrackInfoParams, context: Dict) -> Dict:
    """Get detailed information about a specific Spotify track."""

    token = get_spotify_token()
    if not token:
        return {"error": "Spotify API not configured"}

    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(
            f"https://api.spotify.com/v1/tracks/{params.track_id}",
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        track = response.json()

        return {
            "name": track["name"],
            "artist": ", ".join([artist["name"] for artist in track["artists"]]),
            "album": track["album"]["name"],
            "duration_ms": track["duration_ms"],
            "duration_formatted": f"{track['duration_ms'] // 60000}:{(track['duration_ms'] % 60000) // 1000:02d}",
            "popularity": track.get("popularity", 0),
            "explicit": track.get("explicit", False),
            "url": track["external_urls"]["spotify"],
            "preview_url": track.get("preview_url"),
            "release_date": track["album"].get("release_date", "Unknown"),
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to get track info: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def register_spotify_tools(registry=None):
    """Tools auto-register on import; kept for compatibility."""
    return None
