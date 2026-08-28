import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CHROMA_DIR = "chroma_db"
RAW_DOCS_DIR = "data/raw_docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150