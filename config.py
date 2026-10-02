import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root if present
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen/qwen3.8-27b")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))
DEFAULT_NUM_GENERATIONS = int(os.getenv("DEFAULT_NUM_GENERATIONS", "2"))
DEFAULT_POPULATION_SIZE = int(os.getenv("DEFAULT_POPULATION_SIZE", "3"))

# Tournament defaults
EVALUATION_CRITERIA = [
    "correctness",
    "completeness",
    "reasoning",
    "evidence",
    "clarity",
    "practicality"
]

def reload_keys(groq_key: str = None, tavily_key: str = None) -> None:
    """Dynamically update keys at runtime."""
    global GROQ_API_KEY, TAVILY_API_KEY
    if groq_key:
        GROQ_API_KEY = groq_key.strip()
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    if tavily_key:
        TAVILY_API_KEY = tavily_key.strip()
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def validate_config(groq_key: str = None, tavily_key: str = None) -> dict[str, bool]:
    """Validate presence of necessary API keys."""
    g_key = groq_key or GROQ_API_KEY
    t_key = tavily_key or TAVILY_API_KEY
    return {
        "groq_configured": bool(g_key and not g_key.startswith("...") and len(g_key) > 10),
        "tavily_configured": bool(t_key and not t_key.startswith("...") and len(t_key) > 10)
    }
