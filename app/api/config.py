from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STORAGE_ROOT: str = "./data"

    class Config:
        env_file = ".env"


settings = Settings()
