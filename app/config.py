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
MIN_CHUNK_CHARS = 50  # shorter chunks are page chrome; the shortest useful one (a 2-line Dockerfile) is 53
# Rate limits for /ask (each question is a paid API call). Per visitor IP, plus one shared daily cap.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "100"))
RATE_LIMIT_GLOBAL_PER_DAY = int(os.getenv("RATE_LIMIT_GLOBAL_PER_DAY", "1000"))
# Proxies in front of the app that append to X-Forwarded-For: 0 locally, 1 on Cloud Run
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")