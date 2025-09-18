from pydantic import BaseModel

class TaskContext(BaseModel):
    design_task: str
    domain_description: str