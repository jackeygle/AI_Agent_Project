"""
Tools Module - Define available tools for the Agent
"""
import requests


def search_web(query: str) -> str:
    """Search the web using DuckDuckGo API"""
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get("Abstract"):
            return data["Abstract"]
        elif data.get("RelatedTopics"):
            topics = data["RelatedTopics"][:3]
            results = [t.get("Text", "") for t in topics if t.get("Text")]
            return "\n".join(results) if results else "No results found"
        else:
            return "No results found"
    except Exception as e:
        return f"Search error: {e}"


def get_weather(city: str) -> str:
    """Get weather using wttr.in"""
    try:
        response = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        return response.text.strip()
    except Exception as e:
        return f"Weather error: {e}"


# Tool registry
TOOLS = {
    "search": {
        "description": "Search the web for information. Input: search query",
        "function": search_web
    },
    "weather": {
        "description": "Get current weather for a city. Input: city name",
        "function": get_weather
    }
}


def execute_tool(tool_name: str, tool_input: str) -> str:
    """Execute a tool by name"""
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}"
    return TOOLS[tool_name]["function"](tool_input)


def get_tools_description() -> str:
    """Get formatted description of all tools"""
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"- {name}: {info['description']}")
    return "\n".join(lines)
