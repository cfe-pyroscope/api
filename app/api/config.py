from pydantic_settings import BaseSettings
from pydantic import Extra

class Settings(BaseSettings):
    STORAGE_ROOT: str = "./data"
    class Config:
        env_file = ".env"
        extra = "allow"  # allow unknown env vars like PROJ_LIB. DON'T DELETE


settings = Settings()

