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
    # TODO: allowed type checks
    from onnxconverter_common import data_types as onnx_dtypes
    from .client import DType  # to eliminate circular import
    if prototype is not None:
        if hasattr(prototype, 'shape') and hasattr(prototype, 'dtype'):
            shape = prototype.shape
            dtype = prototype.dtype.name
            mm = {
                'float32': 'FLOAT',
                'float64': 'DOUBLE'}
            if dtype in mm:
                dtype = mm[dtype]
        else:
            raise TypeError(
                "Serializing to ONNX requires shape and dtype"
                " of input data. If `shape` and `dtype` is not passed specifically "
                "it will be inferred from `prototype` "
                "parameter. `prototype` has to be a valid `numpy.ndarray` of shape of your input")
    if not all([shape, dtype]):
        raise RuntimeError((
            "Could not infer shape and dtype. "
            "Either pass them as argument or pass a prototype that has shape and dtype attribute"))
    dtype = dtype.upper()
    if dtype == DType.float32.value:
        onnx_dtype = onnx_dtypes.FloatTensorType(shape)
    elif dtype == DType.float64.value:
        onnx_dtype = onnx_dtypes.DoubleTensorType(shape)
    elif dtype in (DType.int64.value, DType.uint64.value):
        onnx_dtype = onnx_dtypes.Int64TensorType(shape)
    elif dtype in (DType.int32.value, DType.uint32.value): # noqa
        onnx_dtype = onnx_dtypes.Int32TensorType(shape)
    else:
        raise NotImplementedError(
            "'{}' is not supported either by RedisAI or by ONNXRuntime. "
            "You may raise an issue at https://github.com/RedisAI/redisai-py."
            "".format(dtype.lower()))
    return [('input', onnx_dtype)]
