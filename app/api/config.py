from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    STORAGE_ROOT: str = "./data"
    NC_PATH: str = f"{STORAGE_ROOT}/nc"
    ZARR_PATH: str = f"{STORAGE_ROOT}/zarr"

    model_config = {
        "env_file": ".env",
        "extra": "allow" # allow unknown env vars like my PROJ_LIB. DON'T DELETE
    }


settings = Settings()

