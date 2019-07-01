from .version import __version__  # noqa
from .client import (Client, Tensor, BlobTensor, DType, Device, Backend)  # noqa
from .model import (  # noqa
    save_tensorflow, save_torch, save_onnx, save_sklearn, save_sparkml)
from .model import load_model, load_script  # noqa


def save_model(*args, **kwargs):
    raise DeprecationWarning((
        "Single entry for saving different object is deprecated. "
        "Use specific utility for each type. For more information checkout the documentation"))
