from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "http://localhost:3000"

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class _LazySettings:
    """Lazy proxy — loads .env only when accessed, not on import."""

    def __getattr__(self, name: str):
        return getattr(get_settings(), name)


settings = _LazySettings()
