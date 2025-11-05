from pydantic import BaseModel

class Idea(BaseModel):
    text: str