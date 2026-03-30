import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageChops

from svgizer.image_utils import resize_long_side


class Scorer(ABC):
    @abstractmethod
    def prepare_reference(self, original_rgb: Image.Image) -> Any: ...

    @abstractmethod
    def score(self, reference: Any, candidate_png: bytes) -> float: ...

    def diff_heatmap(
        self, reference: Any, candidate_png: bytes, long_side: int = 256
    ) -> bytes | None:
        """Pixel-based diff heatmap (brightness-boosted RGB difference).

        Subclasses may override this with a perceptually richer implementation.
        Returns PNG bytes, or None if the reference lacks an ``image`` attribute.
        """
        ref_img = getattr(reference, "image", None)
        if ref_img is None:
            return None

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        if cand.size != ref_img.size:
            cand = cand.resize(ref_img.size, resample=Image.Resampling.BILINEAR)

        diff = ImageChops.difference(ref_img, cand)
        diff = diff.point(lambda p: min(255, p * 3))

        if long_side > 0:
            diff = resize_long_side(diff, long_side)

        buf = io.BytesIO()
        diff.save(buf, format="PNG")
        return buf.getvalue()


@dataclass(frozen=True)
class ScoreConfig:
    target_long_side: int = 256
    w_vision: float = 0.85
    w_color: float = 0.15


DEFAULT_CONFIG = ScoreConfig()
