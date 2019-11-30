from typing import Union, ByteString, Sequence
from .utils import convert_to_num
from .constants import DType
from .containers import Tensor
try:
    import numpy as np
except (ImportError, ModuleNotFoundError):
    np = None

#  TODO: verify the values for None or invalid values
# TODO: type annotations


def from_numpy(tensor):
    dtype = DType.__members__[str(tensor.dtype)]
    shape = tensor.shape
    blob = bytes(tensor.data)
    return Tensor(blob, shape, dtype, 'BLOB')


def from_sequence(tensor, shape, dtype):
    return Tensor(tensor, shape, dtype, 'VALUES')


def to_numpy(value, shape, dtype):
    # tOdo exception
    dtype = DType.__members__[dtype.lower()].value
    mm = {
        'FLOAT': 'float32',
        'DOUBLE': 'float64'
    }
    if dtype in mm:
        dtype = mm[dtype]
    else:
        dtype = dtype.lower()
    a = np.frombuffer(value, dtype=dtype)
    return a.reshape(shape)


def to_sequence(value, shape, dtype):
    # TODO: what's the need for this? add test cases
    dtype = DType.__members__[dtype.lower()]
    convert_to_num(dtype, value)
    return Tensor(value, tuple(shape), dtype, 'VALUES')
