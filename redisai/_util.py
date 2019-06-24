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


def guess_onnx_dtype(prototype=None, shape=None, dtype=None, input_name='features'):
    # TODO: remove this once this is added to onnxconverter_common
    # TODO: allowed type checks
    from onnxconverter_common import data_types as onnx_dtypes
    from .client import DType  # to eliminate circular import

    if prototype is not None:
        if hasattr(prototype, 'shape') and hasattr(prototype, 'dtype'):
            shape = prototype.shape
            dtype = prototype.dtype.name
        else:
            raise TypeError("`prototype` has to be a valid `numpy.ndarray` of shape of your input")
    if isinstance(dtype, str):
        dtype = DType[dtype.lower()]
    elif not isinstance(dtype, DType):
        raise TypeError("`dtype` has to be of type `numpy.ndarray`/ `str` / `redisai.DType`")
    if not isinstance(shape, tuple) or isinstance(shape, list):
        raise RuntimeError(("Inferred `shape` attribute is not a tuple / list"))

    if dtype == DType.float32:
        onnx_dtype = onnx_dtypes.FloatTensorType(shape)
    elif dtype == DType.float64:
        onnx_dtype = onnx_dtypes.DoubleTensorType(shape)
    elif dtype in (DType.int64, DType.uint64):
        onnx_dtype = onnx_dtypes.Int64TensorType(shape)
    elif dtype in (DType.int32, DType.uint32): # noqa
        onnx_dtype = onnx_dtypes.Int32TensorType(shape)
    else:
        raise NotImplementedError(
            "'{}' is not supported either by RedisAI or by ONNXRuntime. "
            "You may raise an issue at https://github.com/RedisAI/redisai-py."
            "".format(dtype.lower()))
    return [(input_name, onnx_dtype)]
