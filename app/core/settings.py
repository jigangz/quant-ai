"""
Quant-AI Settings - Pydantic Settings with .env support

All configuration is loaded from environment variables.
Use .env file for local development.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== Environment =====
    ENV: Literal["dev", "prod", "test"] = "dev"
    DEBUG: bool = False

    # ===== Database =====
    DATABASE_URL: str = "sqlite:///:memory:"

    # ===== Supabase (optional, for model registry) =====
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    SUPABASE_SERVICE_KEY: str | None = None  # For admin operations

    # ===== Storage (model artifacts) =====
    STORAGE_BACKEND: Literal["local", "supabase", "s3"] = "local"
    STORAGE_LOCAL_PATH: str = "./artifacts"
    S3_BUCKET: str | None = None
    S3_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None

    # ===== Data Providers =====
    PROVIDERS_ENABLED: str = "market"  # Comma-separated: market,sentiment,news
    MARKET_PROVIDER: Literal["yahoo", "polygon"] = "yahoo"
    POLYGON_API_KEY: str | None = None
    NEWS_PROVIDER: Literal["polygon", "mock"] = "mock"
    NEWS_BATCH_SIZE: int = 20

    # ===== Feature Engineering =====
    DEFAULT_FEATURE_GROUPS: str = "ta_basic,volatility"  # Comma-separated

    # ===== Model Training =====
    DEFAULT_MODEL_TYPE: Literal["logistic", "random_forest", "xgboost", "lightgbm"] = (
        "logistic"
    )
    DEFAULT_HORIZON_DAYS: int = 5
    DEFAULT_LABEL_TYPE: Literal["direction", "return"] = "direction"
    TRAINING_MAX_WORKERS: int = 4

    # ===== API =====
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: str = "*"  # Comma-separated origins
    RATE_LIMIT_PER_MINUTE: int = 60
    REQUEST_TIMEOUT_SECONDS: int = 300

    # ===== Redis =====
    REDIS_URL: str = "redis://localhost:6379/0"

    # ===== Infrastructure Backends =====
    BROKER_BACKEND: Literal["kafka", "redis", "memory"] = "redis"
    QUEUE_BACKEND: Literal["sqs", "redis", "memory"] = "redis"
    NOTIFY_BACKEND: Literal["sns", "redis", "memory"] = "redis"
    FUNCTIONS_BACKEND: Literal["lambda", "local"] = "local"

    # ===== Kafka (when BROKER_BACKEND=kafka) =====
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_SASL_USERNAME: str | None = None  # For Upstash / Confluent Cloud / MSK
    KAFKA_SASL_PASSWORD: str | None = None

    # ===== AWS (when using SQS/SNS/Lambda) =====
    AWS_REGION: str = "us-east-1"
    SQS_ENDPOINT_URL: str | None = None  # http://localhost:4566 for LocalStack
    SNS_ENDPOINT_URL: str | None = None  # http://localhost:4566 for LocalStack
    LAMBDA_ENDPOINT_URL: str | None = None  # http://localhost:3001 for SAM CLI

    # ===== Logging =====
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "text"

    # ===== LLM (for V3 RAG) =====
    LLM_PROVIDER: Literal["openai", "anthropic", "mock"] | None = None
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    # ===== Vector Store (for V3 RAG) =====
    VECTOR_BACKEND: Literal["faiss", "pinecone", "qdrant"] | None = None
    PINECONE_API_KEY: str | None = None
    PINECONE_ENVIRONMENT: str | None = None
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    # ===== Helper Methods =====

    @property
    def providers_list(self) -> list[str]:
        """Get enabled providers as list."""
        return [p.strip() for p in self.PROVIDERS_ENABLED.split(",") if p.strip()]

    @property
    def feature_groups_list(self) -> list[str]:
        """Get default feature groups as list."""
        return [g.strip() for g in self.DEFAULT_FEATURE_GROUPS.split(",") if g.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as list."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def is_supabase_configured(self) -> bool:
        """Check if Supabase is configured."""
        return bool(self.SUPABASE_URL and self.SUPABASE_KEY)

    @property
    def is_s3_configured(self) -> bool:
        """Check if S3 is configured."""
        return bool(self.S3_BUCKET and self.AWS_ACCESS_KEY_ID)

    def get_public_settings(self) -> dict:
        """Get settings safe to expose via API (no secrets)."""
        return {
            "env": self.ENV,
            "debug": self.DEBUG,
            "providers_enabled": self.providers_list,
            "market_provider": self.MARKET_PROVIDER,
            "default_feature_groups": self.feature_groups_list,
            "default_model_type": self.DEFAULT_MODEL_TYPE,
            "default_horizon_days": self.DEFAULT_HORIZON_DAYS,
            "storage_backend": self.STORAGE_BACKEND,
            "vector_backend": self.VECTOR_BACKEND,
            "llm_provider": self.LLM_PROVIDER,
            "supabase_configured": self.is_supabase_configured,
            "s3_configured": self.is_s3_configured,
            "broker_backend": self.BROKER_BACKEND,
            "queue_backend": self.QUEUE_BACKEND,
            "notify_backend": self.NOTIFY_BACKEND,
            "functions_backend": self.FUNCTIONS_BACKEND,
        }


# Global settings instance
settings = Settings()
