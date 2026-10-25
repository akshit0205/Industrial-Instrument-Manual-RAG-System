from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    chunk_size: int = 500
    chunk_overlap: int = 100
    model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""
    llm_model: str = "llama3.2"
    mock_llm: bool = False
    manuals_dir: Path = Path("data/manuals")
    index_dir: Path = Path("data/index")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
