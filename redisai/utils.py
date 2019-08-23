import six
from .constants import DType


def to_string(s):
    if isinstance(s, six.string_types):
        return s
    elif isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return s  # Not a string we care about


def str_or_strlist(v):
    if isinstance(v, six.string_types):
        return [v]
    return v


def convert_to_num(dt, arr):
    for ix in six.moves.range(len(arr)):
        obj = arr[ix]
        if isinstance(obj, list):
            convert_to_num(obj)
        else:
            if dt in (DType.float, DType.double):
                arr[ix] = float(obj)
            else:
                arr[ix] = int(obj)
