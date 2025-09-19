import torch
from PIL import Image
import clip

from new_core.interfaces.image_embedder import IImageEmbedder  

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class CLIPImageEmbedder(IImageEmbedder):
    def __init__(self, clip_model: str = "ViT-B/32", device: str | None = None):
        self.device = device or get_device()
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.model.eval()

    def embed(self, image_path: str) -> torch.Tensor:
        """
        Returns a 1D image embedding tensor (CPU) for the given image path.
        """
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device) # type: ignore
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)  # shape (1, D)
        return embedding[0].cpu()
