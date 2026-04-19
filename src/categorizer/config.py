from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    env: str = Field("development", alias="CATEGORIZER_ENV")
    port: int = Field(8000, alias="CATEGORIZER_PORT")
    threads: int = Field(2, alias="CATEGORIZER_THREADS")
    log_level: str = Field("INFO", alias="CATEGORIZER_LOG_LEVEL")

    database_url: PostgresDsn = Field(..., alias="DATABASE_URL")
    # Ops-generated passwords are now URL-encoded by the ansible role, so
    # pydantic's strict RFC-3986 parser accepts them again.
    miplata_ro_database_url: PostgresDsn | None = Field(None, alias="MIPLATA_RO_DATABASE_URL")

    llm_base_url: str = Field("http://llama-server:8080/v1", alias="LLM_BASE_URL")
    llm_model: str = Field("qwen3-4b-instruct-2507", alias="LLM_MODEL")

    api_token: str = Field(..., alias="CATEGORIZER_API_TOKEN")

    rule_min_confidence: float = Field(0.95, alias="CATEGORIZER_RULE_MIN_CONFIDENCE")
    knn_min_confidence: float = Field(0.85, alias="CATEGORIZER_KNN_MIN_CONFIDENCE")
    knn_min_margin: float = Field(0.05, alias="CATEGORIZER_KNN_MIN_MARGIN")
    llm_min_confidence: float = Field(0.70, alias="CATEGORIZER_LLM_MIN_CONFIDENCE")
    think_trigger_confidence: float = Field(0.60, alias="CATEGORIZER_THINK_TRIGGER_CONFIDENCE")

    taxonomy_path: Path = Field(Path("/app/config/taxonomy.yaml"), alias="CATEGORIZER_TAXONOMY_PATH")
    artifacts_dir: Path = Field(Path("/app/artifacts"), alias="CATEGORIZER_ARTIFACTS_DIR")

    # Name of the embedding model used by retrieval.py. fastembed fetches it
    # on first boot and caches under artifacts_dir/embedding_cache/.
    # fastembed 0.8 dropped intfloat/multilingual-e5-small; MiniLM-L12-v2 is
    # the 384-dim multilingual replacement with the same vector dimension, so
    # no pgvector migration is needed.
    embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="CATEGORIZER_EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(384, alias="CATEGORIZER_EMBEDDING_DIM")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
