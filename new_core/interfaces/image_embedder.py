from abc import ABC, abstractmethod
import torch


class IImageEmbedder(ABC):
    @abstractmethod
    def embed(self, image_path: str) -> torch.Tensor:  # can change return type in future if necessary
        ...
