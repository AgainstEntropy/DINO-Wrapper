from PIL import Image
from torchvision import transforms

from .dino.utils import cosine_scheduler


class MaybeToPIL(transforms.ToPILImage):
    """
    Convert a ``tensor`` or ``numpy.ndarray`` to ``PIL Image``, or keep as is if already a ``PIL Image``.
    Args:
        pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to PIL Image.
    Returns:
        Image: Converted image.
    """
    def __call__(self, pic):
        # breakpoint()
        if isinstance(pic, Image.Image):
            return pic
        else:
            return super().__call__(pic)
        