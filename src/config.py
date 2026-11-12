import os
from typing import Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq Cloud API Settings
    GROQ_API_KEY: SecretStr = SecretStr("")
    FAST_LLM_MODEL: str = "llama-3.1-8b-instant"
    SYNTHESIS_LLM_MODEL: str = "llama-3.3-70b-versatile"

    # Qdrant Vector DB Settings
    QDRANT_HOST: Optional[str] = None
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[SecretStr] = None
    QDRANT_COLLECTION: str = "rbi_compliance_hybrid"
    CACHE_COLLECTION: str = "query_semantic_cache"

    def model_post_init(self, __context):
        if self.QDRANT_HOST and not os.getenv("QDRANT_URL"):
            self.QDRANT_URL = f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    # Embedding and Reranking Models
    DENSE_EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_EMBED_MODEL: str = "Qdrant/bm25"
    RERANK_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    RRF_PREFETCH_LIMIT: int = 10
    RERANK_TOP_K: int = 4

    # Agent Guardrails and Thresholds
    SEMANTIC_CACHE_THRESHOLD: float = 0.93
    MAX_AGENT_RETRIES: int = 2

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[SecretStr] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"


settings = Settings()
