from pydantic import BaseModel, Field


class WeatherParams(BaseModel):
    location: str = Field(..., description="City name or location")


def weather_tool(params: WeatherParams, context: dict):
    """Get the current weather for a location."""
    # This is a mock implementation
    locations = {
        "new york": {"temp": 72, "condition": "Sunny"},
        "london": {"temp": 33, "condition": "Rainy"},
        "tokyo": {"temp": 80, "condition": "Clear"},
        "sydney": {"temp": 70, "condition": "Partly Cloudy"},
    }

    location = params.location.lower()
    if location in locations:
        return locations[location]
    else:
        return {"temp": 65, "condition": "Unknown location, using default weather"}


def register_weather_tools(registry):
    """Register weather-related tools with the given registry."""
    registry.register_tool(
        name="weather",
        param_model=WeatherParams,
        function=weather_tool,
        description="Get the current weather for a location",
    )
