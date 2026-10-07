import os
from dotenv import load_dotenv

load_dotenv()

# Resolve paths from the project root so the app works no matter which directory it is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
RAW_DOCS_DIR = os.path.join(BASE_DIR, "data", "raw_docs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")