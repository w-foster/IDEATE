from typing import TypedDict, Dict, Any, NotRequired

class LGModelSpec(TypedDict):
    name: str
    provider: NotRequired[str]
    params: NotRequired[Dict[str, Any]]