

from implicit_nuisance.decoder.base import BaseDecoder
from implicit_nuisance.decoder.image import ImageDecoder
from implicit_nuisance.decoder.linear import LinearDecoder
from implicit_nuisance.decoder.mlp import MLPDecoder


DECODERS = {
    # base classes
    "base": BaseDecoder,
    # backbones
    "image": ImageDecoder,
    "linear": LinearDecoder,
    "mlp": MLPDecoder,
}
__all__ = [
    "base",
    "image",
    "linear",
    "mlp",
]
