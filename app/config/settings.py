from pydantic import BaseSettings
import os

ENV_API = os.getenv("ENVIRONMENT")


class Settings(BaseSettings):
    # user
    secret_key: str
    algorithm: str
    access_token_expires_hours: int

    # clickhouse
    db_ch_host: str
    db_ch_port: int
    db_ch_user: str
    db_ch_protocol: str
    db_ch_database_llma_models: str
    db_ch_table_ai_vniizht_bot: str

    # cv settings
    db_log_processing_video: str
    table_log_processing_video: str
    detection_model_checkpoint_dir: str
    segmentation_model_checkpoint_dir: str

    class Config:
        env_file = ".env" if not ENV_API else f".env.{ENV_API}"


def get_settings():
    return Settings()