from pydantic_settings import BaseSettings, SettingsConfigDict

# config for the application.
class Settings(BaseSettings):
    google_api_key: str
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create an instance
settings = Settings()