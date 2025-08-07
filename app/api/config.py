from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    STORAGE_ROOT: str = "./data"

    model_config = {
        "env_file": ".env",
        "extra": "allow" # allow unknown env vars like my PROJ_LIB. DON'T DELETE
    }


settings = Settings()

