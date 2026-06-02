import os

from dotenv import load_dotenv

load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
EMAIL = os.getenv("EMAIL")
SERPER_URL = os.getenv("SERPER_URL")
