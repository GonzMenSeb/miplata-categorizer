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
    # Kept as plain str so ops-generated passwords with '/' or '+' don't trip
    # pydantic's strict URL parser. SQLAlchemy/psycopg parse at use site.
    miplata_ro_database_url: str | None = Field(None, alias="MIPLATA_RO_DATABASE_URL")

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
    embedding_model: str = Field("intfloat/multilingual-e5-small", alias="CATEGORIZER_EMBEDDING_MODEL")
    embedding_dim: int = Field(384, alias="CATEGORIZER_EMBEDDING_DIM")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
