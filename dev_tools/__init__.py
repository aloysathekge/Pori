# Development tools - not part of core framework
# Delete this directory when done experimenting

from .spotify_tools import register_spotify_tools


def register_dev_tools(registry):
    """Register development/experimental tools"""
    register_spotify_tools(registry)
    print("🎵 Development API tools registered")
