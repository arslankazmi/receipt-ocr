from abc import ABC, abstractmethod
from PIL import Image


class ModelRunner(ABC):
    model_id: str  # short name used in results table

    @abstractmethod
    def extract(self, image: Image.Image) -> dict:
        """Run inference on a single receipt image. Returns a dict (may be partial schema)."""
        ...

    def batch_extract(self, images: list[Image.Image]) -> list[dict]:
        return [self.extract(img) for img in images]
