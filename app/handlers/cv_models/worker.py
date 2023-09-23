import asyncio
import datetime
import uuid
import os

import pandas as pd
import cv2
import pandahouse as ph
from tempfile import NamedTemporaryFile

from app.helpers.db.clickhouse_connector import ClickhouseConnector
from app.config.settings import get_settings
from app.handlers.cv_models.detect_dangers_event import DetectorDangerEvent


settings = get_settings()


class CVWorker:

    def __init__(self, file=None):
        self.file = file

    @staticmethod
    def generate_submit_id() -> str:
        """
        Create new submit
        """
        return str(uuid.uuid4())

    @staticmethod
    async def get_submit_data(submit_id: str):
        """
        Generation file with submit
        TODO: вынести sql в файл - это критическая уязвимость
        """
        async with ClickhouseConnector() as client:
            result = await client.fetch(
                f"""
                    select name_file as filename, count() as cases_count, groupArray(m_s) as timestamps
                    from
                        (
                            select
                                name_file,
                                leftPad(toString(intDivOrZero(seconds, 60)), 2, '0')
                                    || ':' ||
                                leftPad(toString(moduloOrZero(seconds, 60)), 2, '0') as m_s
                            from {settings.db_log_processing_video}.{settings.table_log_processing_video}
                            where submit_id = '{submit_id}' and type_violation in (1, 2)
                            order by name_file, seconds
                        )
                    group by name_file
                """
            )
            if len(result) > 0:
                cols = list(result[0].keys())
                data = pd.DataFrame([dict(zip(cols, el.values())) for el in result])
                return data

    def run_processing_file(self, state_app, submit_id):
        """
        Start processing input video in thread
        """
        video = self.__read_video_file()
        result = self.__compute_predict(video)
        result["submit_id"] = submit_id
        result["name_file"] = self.file.filename
        result["datetime"] = pd.to_datetime(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        asyncio.run(
            self.__save_data_to_clickhouse(
                result,
                settings.db_log_processing_video,
                settings.table_log_processing_video
            )
        )
        state_app.states[submit_id]["count_task_ready"] += 1

    def __read_video_file(self):
        """
        Read video from stream
        """
        temp = NamedTemporaryFile(delete=False)
        try:
            try:
                contents = self.file.file.read()
                with temp as f:
                    f.write(contents)
            except Exception as err:
                return {"message": f"There was an error uploading the file: {err}"}
            finally:
                self.file.file.close()
            res = cv2.VideoCapture(temp.name)
        except Exception as err:
            return {"message": f"There was an error processing the file: {err}"}
        finally:
            os.remove(temp.name)
        return res

    def __compute_predict(self, video):
        detector = DetectorDangerEvent()
        output = detector.predict(video)
        return output

    @staticmethod
    async def __save_data_to_clickhouse(data, db_name, table_name):
        """
        Save data to db
        """
        connection = await ClickhouseConnector(db=db_name).connect()
        result = ph.to_clickhouse(
            data, table_name, index=False, chunksize=100000, connection=connection
        )
        return result

