from pydantic_settings import BaseSettings, SettingsConfigDict

from swapboard.common.network import DEFAULT_API_PORT, DEFAULT_HOST, DEFAULT_UI_PORT


class UISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWAPBOARD_UI_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    api_url: str = f"http://{DEFAULT_HOST}:{DEFAULT_API_PORT}"
    host: str = DEFAULT_HOST
    port: int = DEFAULT_UI_PORT
