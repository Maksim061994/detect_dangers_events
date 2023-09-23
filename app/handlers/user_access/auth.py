import datetime
from fastapi.exceptions import HTTPException
from fastapi import status
from passlib.context import CryptContext
from jose import jwt
from app.handlers.user_access.schemas import UserLoginSchema, UserSchema


DB = {
    "user": UserSchema(login="user", password="$2b$12$8hr6cXforBTbBLeA5DuaNuHrHUM9fQ4fcW0WnBGEE0Jjx7W.5KUae")
}

class UserAccess:
    """
    Управление пользователями
    """

    def __init__(self, settings):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.token_expires_hours = settings.access_token_expires_hours
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    async def login(self, request: UserLoginSchema) -> dict:
        """
        Авторизация пользователя
        :param request: параметры запроса
        :return:
        """
        curr_user = self.__get_user_by_login(request.login)
        if not curr_user:
            raise HTTPException(status_code=403, detail="Пользователь не зарегистрирован в системе")
        if not await self.__validate_password(request.password, curr_user.password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный пароль")
        access_token = await self.__create_access_token(request.dict())
        return {"token": access_token}

    def __get_user_by_login(self, login: str):
        """
        Получение пользователя по логину
        :param login: логин
        :return: пользователь
        """
        return DB.get(login)

    async def __validate_password(self, request_pass, user_pass):
        """
        Проверка валидности пароля
        :param request_pass: пароль из запроса
        :param user_pass: пароль из БД
        :return: результат проверки
        """
        return self.pwd_context.verify(request_pass, user_pass)

    async def __create_access_token(self, data: dict):
        """
        Создание токена для пользователя
        :param data: данные для создания токена
        :return: токен
        """
        expires = datetime.timedelta(hours=int(self.token_expires_hours))
        if self.token_expires_hours:
            expires = datetime.datetime.now() + expires
        expires_timestamp = expires.timestamp()
        to_encode = data.copy()
        to_encode.update({"exp": expires_timestamp})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
