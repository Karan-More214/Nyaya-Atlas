"""Central configuration for NyayaAtlas."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# Where your source documents (PDF / TXT) live
DATA_DIR = ROOT / "data" / "raw"

# Where the search index is stored
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
COLLECTION_NAME = "nyayaatlas"

# Embeddings (runs locally, free)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking (done per page so every chunk maps to exactly one page number)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "250"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
CANDIDATES = int(os.getenv("CANDIDATES", "20"))  # fetched from each retriever before fusion
RRF_K = 60  # reciprocal rank fusion constant

# Abstention: if the best vector similarity is below this, answer "not found".
# Defaults below were chosen with eval/sweep.py (see README, Evaluation).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.50"))

# Optional cross-encoder reranker (more accurate, slower, bigger download)
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM (optional - without a key the app shows the retrieved passages only)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
