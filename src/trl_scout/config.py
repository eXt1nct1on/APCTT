"""Application settings loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for TRL-Scout."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "provider/model-name"
    llm_fallback_model: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = ""
    data_dir: str = "data"
    max_parallel_llm: int = 4
    max_upload_mb: int = 25
    no_cache: bool = False


settings = Settings()


class ScoringConfig(BaseSettings):
    """Tunable thresholds and factors for TRL scoring."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SCORING_",
        extra="ignore",
    )

    rule_threshold: float = Field(0.60, ge=0, le=1)
    evidence_jump_threshold: float = Field(0.30, ge=0, le=1)
    range_partial_low: float = Field(0.35, ge=0, le=1)
    range_partial_high: float = Field(0.60, ge=0, le=1)
    cited_only_cap: float = Field(0.50, ge=0, le=1)
    hypothetical_only_cap: float = Field(0.25, ge=0, le=1)
    confidence_spread_divisor: float = Field(4.0, gt=0)
    evidence_factor_cap: int = Field(6, gt=0)
    patent_no_examples_factor: float = Field(0.80, ge=0, le=1)
    ocr_factor: float = Field(0.90, ge=0, le=1)
    ocr_page_fraction: float = Field(0.30, ge=0, le=1)


scoring_config = ScoringConfig()
