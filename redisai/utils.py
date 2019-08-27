import six
from .constants import DType


def to_string(s):
    if isinstance(s, six.string_types):
        return s
    elif isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return s  # Not a string we care about


def str_or_strsequence(v):
    if not isinstance(v, (list, tuple)):
        if isinstance(v, six.string_types):
            return [v]
        else:
            raise TypeError('Argument must be a string, list or a tuple')
    return v


def convert_to_num(dt: DType, arr) -> None:
    """
    Recurse value, replacing each element of b'' with the appropriate element.
    Function doesn't return anything but does inplace operation which updates `arr`

    :param dt: Type of tensor | array
    :param arr: The array with b'' numbers or recursive array of b''
    """
    for ix in six.moves.range(len(arr)):
        obj = arr[ix]
        if isinstance(obj, list):
            convert_to_num(dt, obj)
        else:
            if dt in (DType.float, DType.double):
                arr[ix] = float(obj)
            else:
                arr[ix] = int(obj)
