from pydantic import BaseModel

class TaskGuardrails(BaseModel):
    text: str