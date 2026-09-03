from pydantic_settings import BaseSettings, SettingsConfigDict


class UISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWAPBOARD_UI_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    api_url: str = "http://127.0.0.1:8771"
    host: str = "127.0.0.1"
    port: int = 8770
