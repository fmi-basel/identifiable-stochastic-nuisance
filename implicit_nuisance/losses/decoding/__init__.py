
from torch.nn import functional as F

__all__ = [
    "mse",
    "cross_entropy",
]

DECODER_LOSS_FUNCTIONS = {
    "mse": F.mse_loss,
    "cross_entropy": F.cross_entropy,
}