from typing import TypedDict, Dict, Any, NotRequired

class LGModelSpec(TypedDict, total=False):
    model: str
    provider: NotRequired[str]
    params: NotRequired[Dict[str, Any]]