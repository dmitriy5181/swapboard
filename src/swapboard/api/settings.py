from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from swapboard.common.network import DEFAULT_LLAMA_SWAP_PORT
from swapboard.common.paths import Layout


class Settings(BaseSettings):
    """Where the API reads models from when nothing overrides it.

    The paths default to the layout a deployment owns, resolved from the
    virtualenv this process runs in, so an installed swapboard finds its own
    config and models without being told where they are.
    """

    model_config = SettingsConfigDict(
        env_prefix="SWAPBOARD_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    llama_swap_config_path: str = Field(
        default_factory=lambda: str(Layout.default().llama_swap_config)
    )
    llama_swap_port: int = DEFAULT_LLAMA_SWAP_PORT
    models_path: str = Field(default_factory=lambda: str(Layout.default().models))
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
