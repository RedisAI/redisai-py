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


def infer_shape_dtype(prototype, shape, dtype):
    # TODO: perhaps move this to onnxconverter_common
    from .client import DType  # to eliminate circular import
    if prototype is not None:
        if hasattr(prototype, 'shape') and hasattr(prototype, 'dtype'):
            shape = prototype.shape
            dtype = prototype.dtype.name
        else:
            raise TypeError("`prototype` has to be a valid `numpy.ndarray` of shape of your input")
    if isinstance(dtype, str):
        try:
            dtype = DType[dtype.lower()]
        except KeyError:
            raise RuntimeError(
                f"{dtype} is not supported by RedisAI "
                "You may raise an issue at https://github.com/RedisAI/redisai-py.")
    elif not isinstance(dtype, DType):
        raise TypeError("`dtype` has to be of type `numpy.ndarray`/ `str` / `redisai.DType`")
    if not isinstance(shape, tuple) or isinstance(shape, list):
        raise RuntimeError(("Inferred `shape` attribute is not a tuple / list"))
    return shape, dtype


def guess_onnx_tensortype(shape, dtype, input_name='features'):
    # TODO: remove this once this is added to onnxconverter_common or move this there
    from onnxconverter_common import data_types as onnx_dtypes
    from .client import DType  # to eliminate circular import

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
            f"'{dtype.name}' is not supported by ONNXRuntime. "
            "You may raise an issue at https://github.com/RedisAI/redisai-py.")
    return [(input_name, onnx_dtype)]
