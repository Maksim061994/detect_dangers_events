from pydantic import BaseModel, constr


class UserLoginSchema(BaseModel):
    login: constr(min_length=4, max_length=16)
    password: str


class UserSchema(BaseModel):
    login: str
    password: str
