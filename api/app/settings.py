"""Environment-backed service configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings."""

    model_config = SettingsConfigDict(env_prefix="PKR_", env_file=".env", extra="ignore")

    model_dir: Path = Path("artifacts/models/current")
    classifier_path: Path = Path("artifacts/models/current/classifier.onnx")
    model_manifest: Path = Path("artifacts/models/current/manifest.json")
    max_upload_bytes: int = Field(default=12 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
    max_concurrent_inferences: int = Field(default=2, gt=0, le=32)
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
