from jose.exceptions import ExpiredSignatureError, JWTError
from fastapi.exceptions import HTTPException
from jose import jwt
from app.config.settings import get_settings
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends

settings = get_settings()

SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
TOKEN_EXPIRES_HOURS = settings.access_token_expires_hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Время жизни токена истекло")
    except JWTError:
        raise HTTPException(status_code=406, detail="Невалидный токен")
