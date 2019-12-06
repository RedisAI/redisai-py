from typing import Union, ByteString, Sequence
from .utils import convert_to_num
from .constants import DType
from .containers import Tensor
try:
    import numpy as np
except (ImportError, ModuleNotFoundError):
    np = None


def from_numpy(tensor: np.ndarray) -> Tensor:
    """ Convert the numpy input from user to `Tensor` """
    dtype = DType.__members__[str(tensor.dtype)]
    shape = tensor.shape
    blob = bytes(tensor.data)
    return Tensor(blob, shape, dtype, 'BLOB')


def from_sequence(tensor: Sequence, shape: Union[list, tuple], dtype: DType) -> Tensor:
    """ Convert the `list`/`tuple` input from user to `Tensor` """
    return Tensor(tensor, shape, dtype, 'VALUES')


def to_numpy(value: ByteString, shape: Union[list, tuple], dtype: DType) -> np.ndarray:
    """ Convert `BLOB` result from RedisAI to `np.ndarray` """
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


def to_sequence(value: list, shape: list, dtype: DType) -> Tensor:
    """ Convert `VALUES` result from RedisAI to `Tensor` """
    dtype = DType.__members__[dtype.lower()]
    convert_to_num(dtype, value)
    return Tensor(value, tuple(shape), dtype, 'VALUES')
