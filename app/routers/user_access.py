from fastapi import APIRouter

# handlers
from app.handlers.user_access.schemas import UserLoginSchema
from app.handlers.user_access.auth import UserAccess
from app.config.settings import get_settings


user_access_router = APIRouter()
settings = get_settings()


@user_access_router.post('/login')
async def login(request: UserLoginSchema):
    user = UserAccess(settings)
    return await user.login(request)
