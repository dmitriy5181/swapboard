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
    # 8770 is deliberately avoided: macOS runs com.apple.sharingd there for
    # Continuity and AirDrop, so binding it fails on any Mac.
    port: int = 8773
