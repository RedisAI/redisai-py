from .version import __version__  # noqa
from .client import (Client, Tensor, BlobTensor, DType, Device, Backend)  # noqa
from .model import (  # noqa
    save_tensorflow, save_torch, save_onnx, save_sklearn, save_sparkml,
    save_coreml, save_libsvm, save_lightgbm, save_xgboost)
from .model import load_model  # noqa


def save_model(*args, **kwargs):
    """
    Deprecated
    """
    from .model import Model
    Model.save(*args, **kwargs)
