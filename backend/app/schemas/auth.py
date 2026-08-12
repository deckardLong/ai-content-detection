from pydantic import BaseModel, Field

# User Create
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6)

# User Login
class UserLogin(BaseModel):
    username: str
    password: str

# User Out
class UserOut(BaseModel):
    id: str
    username: str
    avatar_url: str | None = None

# Token
class Token(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    user: UserOut