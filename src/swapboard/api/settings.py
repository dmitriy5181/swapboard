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
    public_endpoint_url: str | None = None

    @field_validator("hf_token", "public_endpoint_url", mode="before")
    @classmethod
    def normalize_blank(cls, value: object) -> object:
        """Treats an empty value as absent.

        Deployment templates commonly leave these variables defined but blank.
        A blank token would make huggingface_hub send an empty Authorization
        header instead of falling back to anonymous access, and a blank
        endpoint would hide the derived one behind an empty string.
        """
        if isinstance(value, str):
            return value.strip() or None
        return value
