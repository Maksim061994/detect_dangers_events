from fastapi import APIRouter, HTTPException, status, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from http import HTTPStatus
import io
from app.handlers.cv_models.worker import CVWorker

cv_router = APIRouter()


class TaskState:

    def __init__(self):
        self.states = {}

    def get_state(self, submit_id: str):
        status = self.states.get(submit_id)
        if status is None:
            return status
        return self.states.get("count_task_queue") == self.states.get("count_task_ready")


state = TaskState()


@cv_router.post('/cv/video/submit')
async def cv_video_create_submit():
    worker = CVWorker()
    submit_id = worker.generate_submit_id()
    state.states[submit_id] = {}
    state.states[submit_id]["count_task_queue"] = 0
    state.states[submit_id]["count_task_ready"] = 0
    return {"submit_id": submit_id}


@cv_router.get('/cv/video/submit')
async def cv_video_generate_submit(submit_id: str):
    # if submit_id not in state.states:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail=f"Данный submit не найден. Сначала необходимо создать submit",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )
    # if state.get_state(submit_id) is False:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail=f"Для данного submit еще идет обработка видео",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )
    worker = CVWorker()
    result = await worker.get_submit_data(submit_id=submit_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Данный submit не найден. Сначала необходимо создать submit",
            headers={"WWW-Authenticate": "Bearer"},
        )
    stream = io.StringIO()
    result.to_csv(stream, index = False)
    response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv"
    )
    response.headers["Content-Disposition"] = f"attachment; filename=submission_{submit_id}.csv"
    return response


@cv_router.post('/cv/video/upload', status_code=HTTPStatus.ACCEPTED)
def сv_video_upload(submit_id: str, file: UploadFile):
    # if submit_id not in state.states:
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail=f"Сначала необходимо создать submit",
    #         headers={"WWW-Authenticate": "Bearer"},
    #     )
    if not file.filename.endswith(".mp4"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Сервис работает только с mp4 файлами",
            headers={"WWW-Authenticate": "Bearer"},
        )
    worker = CVWorker(file)
    worker.run_processing_file(state_app=state, submit_id=submit_id)  # TODO: пока синхронно - потом добавить создание задач в очереди
    state.states[submit_id]["count_task_queue"] += 1  # можно использовать в будущем для очередей, но лучше Rabbit
    return {"msg": f"video upload with - {submit_id}"}

