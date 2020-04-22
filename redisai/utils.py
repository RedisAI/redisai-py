from typing import Union, ByteString, Sequence
import numpy as np


dtype_dict = {
    'float': 'FLOAT',
    'double': 'DOUBLE',
    'float32': 'FLOAT',
    'float64': 'DOUBLE',
    'int8': 'INT8',
    'int16': 'INT16',
    'int32': 'INT32',
    'int64': 'INT64',
    'uint8': 'UINT8',
    'uint16': 'UINT16',
    'uint32': 'UINT32',
    'uint64': 'UINT64'}


def numpy2blob(tensor: np.ndarray) -> tuple:
    """ Convert the numpy input from user to `Tensor` """
    try:
        dtype = dtype_dict[str(tensor.dtype)]
    except KeyError:
        raise TypeError(f"RedisAI doesn't support tensors of type {tensor.dtype}")
    shape = tensor.shape
    blob = bytes(tensor.data)
    return dtype, shape, blob


def blob2numpy(value: ByteString, shape: Union[list, tuple], dtype: str) -> np.ndarray:
    """ Convert `BLOB` result from RedisAI to `np.ndarray` """
    mm = {
        'FLOAT': 'float32',
        'DOUBLE': 'float64'
    }
    dtype = mm.get(dtype, dtype.lower())
    a = np.frombuffer(value, dtype=dtype)
    return a.reshape(shape)


def list2dict(lst):
    if len(lst) % 2 != 0:
        raise RuntimeError("Can't unpack the list: {}".format(lst))
    out = {}
    for i in range(0, len(lst), 2):
        key = lst[i].decode().lower()
        val = lst[i + 1]
        if key != 'blob' and isinstance(val, bytes):
            val = val.decode()
        out[key] = val
    return out


def un_bytize(arr: list, target_type: type) -> list:
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


def listify(inp: Union[str, Sequence[str]]) -> Sequence[str]:
    return (inp,) if not isinstance(inp, (list, tuple)) else inp

