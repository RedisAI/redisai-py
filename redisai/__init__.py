from .version import __version__
from .client import (Client, Tensor, BlobTensor, DType, Device, Backend)


def save_model(*args, **kwargs):
    """
    Importing inside to avoid loading the TF/PyTorch/ONNX
    into the scope unnecessary. This function wraps the
    internal save model utility to make it user friendly
    """
    from .model import Model
    Model.save(*args, **kwargs)


def load_model(*args, **kwargs):
    """
    Importing inside to avoid loading the TF/PyTorch/ONNX
    into the scope unnecessary. This function wraps the
    internal load model utility to make it user friendly
    """
    from .model import Model
    return Model.load(*args, **kwargs)
