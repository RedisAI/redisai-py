from typing import Union, ByteString, Sequence
import numpy as np
from enum import Enum


class DType(Enum):
    float = 'FLOAT'
    double = 'DOUBLE'
    int8 = 'INT8'
    int16 = 'INT16'
    int32 = 'INT32'
    int64 = 'INT64'
    uint8 = 'UINT8'
    uint16 = 'UINT16'
    uint32 = 'UINT32'
    uint64 = 'UINT64'
    # aliases
    float32 = 'FLOAT'
    float64 = 'DOUBLE'


def numpy2blob(tensor: np.ndarray) -> tuple:
    """ Convert the numpy input from user to `Tensor` """
    # TODO: May be change the DTYPE enum
    dtype = DType.__members__[str(tensor.dtype)].value
    shape = tensor.shape
    blob = bytes(tensor.data)
    return dtype, shape, blob


def blob2numpy(value: ByteString, shape: Union[list, tuple], dtype: DType) -> np.ndarray:
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


def list2dict(lst):
    if len(lst) % 2 != 0:
        raise RuntimeError("Can't unpack the list: {}".format(lst))
    out = {}
    for i in range(0, len(lst), 2):
        val = lst[i + 1]
        if isinstance(val, bytes):
            val = val.decode()
        out[lst[i].decode().lower()] = val
    return out


def un_bytize(arr: Sequence, target_type: type) -> Sequence:
    """
    Recurse value, replacing each element of b'' with the appropriate element.
    Function returns the same array after inplace operation which updates `arr`

    :param target_type: Type of tensor | array
    :param arr: The array with b'' numbers or recursive array of b''
    """
    for ix in range(len(arr)):
        obj = arr[ix]
        if isinstance(obj, list):
            un_bytize(obj, target_type)
        else:
            arr[ix] = target_type(obj)
    return arr
