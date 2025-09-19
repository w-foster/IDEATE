from pydantic import BaseModel

class TaskConstraints(BaseModel):
    text: str