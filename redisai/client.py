from enum import Enum
from redis import StrictRedis
from ._util import to_string
import six

try:
    import numpy as np
except ImportError:
    np = None

try:
    from typing import Union, Any, AnyStr, ByteString, Collection, Type
except ImportError:
    pass


class Device(Enum):
    cpu = 'cpu'
    gpu = 'gpu'


class Backend(Enum):
    tf = 'tf'
    torch = 'torch'
    onnx = 'ort'


class DType(Enum):
    float = 'float'
    double = 'double'
    int8 = 'int8'
    int16 = 'int16'
    int32 = 'int32'
    int64 = 'int64'
    uint8 = 'uint8'
    uint16 = 'uint16'
    uint32 = 'uint32'
    uint64 = 'uint64'

    # aliases
    float32 = 'float'
    float64 = 'double'


def _str_or_strlist(v):
    if isinstance(v, six.string_types):
        return [v]
    return v


def _convert_to_num(dt, arr):
    for ix in six.moves.range(len(arr)):
        obj = arr[ix]
        if isinstance(obj, list):
            _convert_to_num(obj)
        else:
            if dt in (DType.float, DType.double):
                arr[ix] = float(obj)
            else:
                arr[ix] = int(obj)


class Tensor(object):
    ARGNAME = 'VALUES'

    def __init__(self,
                 dtype,  # type: DType
                 shape,  # type: Collection[int]
                 value):
        """
        Declare a tensor suitable for passing to tensorset
        :param dtype: The type the values should be stored as.
            This can be one of Tensor.FLOAT, tensor.DOUBLE, etc.
        :param shape: An array describing the shape of the tensor. For an
            image 250x250 with three channels, this would be [250, 250, 3]
        :param value: The value for the tensor. Can be an array.
            The contents must coordinate with the shape, meaning that the
            overall length needs to be the product of all figures in the
            shape. There is no verification to ensure that each dimension
            is correct. Your application must ensure that the ordering
            is always consistent.
        """
        self.type = dtype
        self.shape = list(shape)
        self.value = value
        if not isinstance(value, (list, tuple)):
            self.value = [value]

    def __repr__(self):
        return '<{c.__class__.__name__}(shape={s} type={t}) at 0x{id:x}>'.format(
            c=self,
            s=self.shape,
            t=self.type,
            id=id(self))

    @classmethod
    def from_resp(cls, dtype, shape, value):
        # recurse value, replacing each element of b'' with the
        # appropriate element
        _convert_to_num(dtype, value)
        return cls(dtype, shape, value)

    @classmethod
    def scalar(cls, dtype, *items):
        """
        Create a tensor from a list of numbers
        :param dtype: Type to use for storage
        :param items: One or more items
        :return: Tensor
        """
        return cls(dtype, [len(items)], items)


class BlobTensor(Tensor):
    ARGNAME = 'BLOB'

    def __init__(self,
                 dtype,
                 shape,  # type: Collection[int]
                 *blobs  # type: Union[BlobTensor, ByteString]
                 ):
        """
        Create a tensor from a binary blob
        :param dtype: The datatype, one of Tensor.FLOAT, Tensor.DOUBLE, etc.
        :param shape: An array
        :param blobs: One or more blobs to assign to the tensor.
        """
        if len(blobs) > 1:
            blobarr = bytearray()
            for b in blobs:
                if isinstance(b, BlobTensor):
                    b = b.value[0]
                blobarr += b
            size = len(blobs)
            blobs = bytes(blobarr)
            shape = [size] + list(shape)
        else:
            blobs = bytes(blobs[0])

        super(BlobTensor, self).__init__(dtype, shape, blobs)

    @classmethod
    def from_numpy(cls, *nparrs):
        # type: (type, np.array) -> BlobTensor
        blobs = []
        for arr in nparrs:
            blobs.append(arr.data)
        dt = DType.__members__[str(nparrs[0].dtype)]
        return cls(dt, nparrs[0].shape, *blobs)

    @property
    def blob(self):
        return self.value[0]

    def to_numpy(self):
        # type: () -> np.array
        a = np.frombuffer(self.value[0], dtype=self._to_numpy_type(self.type))
        return a.reshape(self.shape)

    @staticmethod
    def _to_numpy_type(t):
        t = t.lower()
        mm = {
            'float': 'float32',
            'double': 'float64'
        }
        if t in mm:
            return mm[t]
        return t

    @classmethod
    def from_resp(cls, dtype, shape, value):
        return cls(dtype, shape, value)


class Client(StrictRedis):
    def modelset(self,
                 name,  # type: AnyStr
                 backend,  # type: Backend
                 device,  # type: Device
                 data,  # type: ByteString
                 input=None,  # type: Union[AnyStr|Collection[AnyStr]|None]
                 output=None  # type: Union[AnyStr|Collection[AnyStr]|None]
                 ):
        args = ['AI.MODELSET', name, backend.value, device.value]
        if backend == Backend.tf:
            if not(all((input, output))):
                raise ValueError(
                    'Require keyword arguments input and output for TF models')
            args += ['INPUTS'] + _str_or_strlist(input)
            args += ['OUTPUTS'] + _str_or_strlist(output)
        args += [data]
        return self.execute_command(*args)

    def modelget(self, name):
        rv = self.execute_command('AI.MODELGET', name)
        return {
            'backend': Backend(rv[0]),
            'device': Device(rv[1]),
            'data': rv[2]
        }

    def modelrun(self,
                 name,
                 input,  # type: Union[AnyStr|Collection[AnyStr]]
                 output  # type: Union[AnyStr|Collection[AnyStr]]
                 ):
        args = ['AI.MODELRUN', name]
        args += ['INPUTS'] + _str_or_strlist(input)
        args += ['OUTPUTS'] + _str_or_strlist(output)
        return self.execute_command(*args)

    def tensorset(self, key, tensor):
        # type: (Client, AnyStr, Union[Tensor, np.ndarray]) -> Any
        """
        Set the values of the tensor on the server using the provided Tensor object
        :param key: The name of the tensor
        :param tensor: a `Tensor` object
        """
        if np and isinstance(tensor, np.ndarray):
            tensor = BlobTensor.from_numpy(tensor)
        args = ['AI.TENSORSET', key, tensor.type.value]
        args += tensor.shape
        args += [tensor.ARGNAME]
        args += tensor.value
        return self.execute_command(*args)

    def tensorget(self, key, as_type=Tensor, meta_only=False):
        # type: (AnyStr, Type[Tensor], bool) -> Tensor
        """
        Retrieve the value of a tensor from the server
        :param key: the name of the tensor
        :param as_type: the resultant tensor type
        :param meta_only: if true, then the value is not retrieved,
            only the shape and the type
        :return: an instance of as_type
        """
        argname = 'META' if meta_only else as_type.ARGNAME
        res = self.execute_command('AI.TENSORGET', key, argname)
        dtype, shape = to_string(res[0]), res[1]
        if meta_only:
            return as_type(dtype, shape, [])
        else:
            return as_type.from_resp(dtype, shape, res[2])

    def scriptset(self, name, device, script):
        return self.execute_command('AI.SCRIPTSET', name, device.value, script)

    def scriptget(self, name):
        r = self.execute_command('AI.SCRIPTGET', name)
        return {
            'device': to_string(r[0]),
            'script': to_string(r[1])
        }

    def scriptrun(self,
                  name,
                  function,  # type: AnyStr
                  input,  # type: Union[AnyStr|Collection[AnyStr]]
                  output  # type: Union[AnyStr|Collection[AnyStr]]
                  ):
        args = ['AI.SCRIPTRUN', name, function, 'INPUTS']
        args += _str_or_strlist(input)
        args += ['OUTPUTS']
        args += _str_or_strlist(output)
        return self.execute_command(*args)
