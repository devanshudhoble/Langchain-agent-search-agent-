import os
from pathlib import Path
from dotenv import load_dotenv

def load_env():
    """Load .env file to expose GEMINI_API_KEY."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
