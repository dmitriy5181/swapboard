from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWAPBOARD_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    llama_swap_config_path: str = "/etc/llama-swap/config/config.yaml"
    llama_swap_port: int = 8080
    models_path: str = "/models"
    hf_token: str | None = None

    @field_validator("hf_token", mode="before")
    @classmethod
    def normalize_blank_hf_token(cls, value: object) -> object:
        """Treats an empty token as absent.

        Deployment templates commonly leave the variable defined but blank, and
        passing that through would make huggingface_hub send an empty
        Authorization header instead of falling back to anonymous access.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value
