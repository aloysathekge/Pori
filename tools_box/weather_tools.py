class WeatherParams(BaseModel):
    location: str = Field(..., description="City name or location")


def weather_tool(params: WeatherParams, context: dict):
    """Get the current weather for a location."""
    # This is a mock implementation
    locations = {
        "new york": {"temp": 72, "condition": "Sunny"},
        "london": {"temp": 62, "condition": "Rainy"},
        "tokyo": {"temp": 80, "condition": "Clear"},
        "sydney": {"temp": 70, "condition": "Partly Cloudy"},
    }

    location = params.location.lower()
    if location in locations:
        return locations[location]
    else:
        return {"temp": 65, "condition": "Unknown location, using default weather"}
