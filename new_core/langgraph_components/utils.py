from langgraphs.types import LGModelSpec
from new_core.models.ai_model_spec import AIModelSpec


def to_langgraph_spec(model_spec: AIModelSpec) -> LGModelSpec:
    return LGModelSpec(
        name=model_spec.name,
        provider=model_spec.provider,
        params=model_spec.params
    )