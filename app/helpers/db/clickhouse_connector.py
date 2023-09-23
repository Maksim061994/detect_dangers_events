from aiochclient import ChClient
from aiohttp import ClientSession
from app.config.settings import get_settings

settings = get_settings()


class ClickhouseConnector:

    def __init__(self, db=None):
        self.url_ch = f"{settings.db_ch_protocol}://{settings.db_ch_host}:{settings.db_ch_port}"
        self.connection_ch = dict(database=db, host=self.url_ch, user='default', password='')
        self.client = ChClient(
            session=ClientSession(),
            url=self.url_ch
        )

    async def connect(self):
        return self.connection_ch

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

