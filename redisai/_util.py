import six
import importlib


def to_string(s):
    if isinstance(s, six.string_types):
        return s
    elif isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return s  # Not a string we care about


def is_installed(packages):
    if not isinstance(packages, list):
        packages = [packages]
    for p in packages:
        if importlib.util.find_spec(p) is None:
            return False
    return True


def guess_onnx_dtype(prototype=None, shape=None, dtype=None):
    # TODO: remove this once this is added to onnxconverter_common
    from onnxconverter_common import data_types as onnx_dtypes
    import numpy as np
    if prototype:
        if hasattr(prototype, 'shape') and hasattr(prototype, 'dtype'):
            shape = prototype.shape
            dtype = prototype.dtype
        else:
            raise TypeError(
                "Serializing to ONNX requires shape and dtype"
                " of input data. If `shape` and `dtype` is not passed specifically "
                "it will be inferred from `prototype` "
                "parameter. `prototype` has to be a valid `numpy.ndarray` of shape of your input")
    if not all(shape, dtype):
        raise RuntimeError((
            "Could not infer shape and dtype. "
            "Either pass them as argument or pass a prototype that has shape and dtype attribute"))
    if dtype == np.float32:
        return onnx_dtypes.FloatTensorType(shape)
    elif dtype in (np.str, str, object) or str(
        dtype) in ('<U1', ): # noqa
        return onnx_dtypes.StringTensorType(shape)
    elif dtype in (np.int64, np.uint64) or str(dtype) == '<U6':
        return onnx_dtypes.Int64TensorType(shape)
    elif dtype in (np.int32, np.uint32) or str(
        dtype) in ('<U4', ): # noqa
        return onnx_dtypes.Int32TensorType(shape)
    elif dtype == np.bool:
        return onnx_dtypes.BooleanTensorType(shape)
    else:
        raise NotImplementedError(
            "Unsupported dtype '{}'. You may raise an issue "
            "at https://github.com/RedisAI/redisai-py."
            "".format(dtype))
