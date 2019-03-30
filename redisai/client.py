from redis import StrictRedis
from ._util import to_string

try:
    import numpy as np
except ImportError:
    np = None

try:
    from typing import Union, Any, AnyStr, ByteString, Collection
except ImportError:
    pass


DEVICE_CPU = 'cpu'
DEVICE_GPU = 'gpu'

BACKEND_TF = 'tf'
BACKEND_TORCH = 'torch'
BACKEND_ONNX = 'ort'


class Tensor(object):
    FLOAT = 'float'
    DOUBLE = 'double'
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'

    ARGNAME = 'VALUES'

    def __init__(self,
                 ttype,  # type: AnyStr
                 shape,  # type: Collection[int]
                 value):
        """
        Declare a tensor suitable for passing to tensorset
        :param ttype: The type the values should be stored as.
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
        self.type = ttype
        self.shape = shape
        self.value = value
        self._size = 1
        if not isinstance(value, (list, tuple)):
            self.value = [value]

    @property
    def size(self):
        return self._size

    def __repr__(self):
        return '<{c.__class__.__name__}(shape={s} type={t}) at 0x{id:x}>'.format(
            c=self,
            s=self.shape,
            t=self.type,
            id=id(self))


class ScalarTensor(Tensor):
    def __init__(self, dtype, *values):
        # type: (ScalarTensor, AnyStr, Any) -> None
        """
        Declare a tensor with a bunch of scalar values. This can be used
        to 'batch-load' several tensors.

        :param dtype: The datatype to store the tensor as
        :param values: List of values
        """
        super(ScalarTensor, self).__init__(dtype, [1], values)
        self._size = len(values)


class BlobTensor(Tensor):
    ARGNAME = 'BLOB'

    def __init__(self,
                 ttype,
                 shape,  # type: Collection[int]
                 *blobs  # type: Union[BlobTensor, ByteString]
                 ):
        """
        Create a tensor from a binary blob
        :param ttype: The datatype, one of Tensor.FLOAT, Tensor.DOUBLE, etc.
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
        else:
            blobs = bytes(blobs[0])
            size = 1

        super(BlobTensor, self).__init__(ttype, shape, blobs)
        self._size = size

    @classmethod
    def from_numpy(cls, *nparrs):
        # type: (type, np.array) -> BlobTensor
        blobs = []
        for arr in nparrs:
            blobs.append(arr.data)
        return cls(
            BlobTensor._from_numpy_type(nparrs[0].dtype),
            nparrs[0].shape, *blobs)

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

    @staticmethod
    def _from_numpy_type(t):
        t = str(t).lower()
        mm = {
            'float32': 'float',
            'float64': 'double',
            'float_': 'double'
        }
        if t in mm:
            return mm[t]
        return t


class Client(StrictRedis):
    def modelset(self, name, backend, device, inputs, outputs, data):
        args = ['AI.MODELSET', name, backend, device, 'INPUTS']
        args += inputs
        args += ['OUTPUTS'] + outputs
        args += [data]
        return self.execute_command(*args)

    def modelget(self, name):
        rv = self.execute_command('AI.MODELGET', name)
        return {
            'backend': rv[0],
            'device': rv[1],
            'data': rv[2]
        }

    def modelrun(self, name, inputs, outputs):
        args = ['AI.MODELRUN', name]
        args += ['INPUTS'] + inputs + ['OUTPUTS'] + outputs
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
        args = ['AI.TENSORSET', key, tensor.type, tensor.size]
        args += tensor.shape
        args += [tensor.ARGNAME]
        args += tensor.value
        return self.execute_command(*args)

    def tensorget(self, key, astype=Tensor, meta_only=False):
        """
        Retrieve the value of a tensor from the server
        :param key: the name of the tensor
        :param astype: the resultant tensor type
        :param meta_only: if true, then the value is not retrieved,
            only the shape and the type
        :return: an instance of astype
        """
        argname = 'META' if meta_only else astype.ARGNAME
        res = self.execute_command('AI.TENSORGET', key, argname)
        dtype, shape = to_string(res[0]), res[1]
        if meta_only:
            return astype(dtype, shape, [])
        else:
            return astype(dtype, shape, res[2])

    def scriptset(self, name, device, script):
        return self.execute_command('AI.SCRIPTSET', name, device, script)

    def scriptget(self, name):
        r = self.execute_command('AI.SCRIPTGET', name)
        return {
            'device': to_string(r[0]),
            'script': to_string(r[1])
        }

    def scriptrun(self, name, function, inputs, outputs):
        args = ['AI.SCRIPTRUN', name, function, 'INPUTS']
        args += inputs
        args += ['OUTPUTS']
        args += outputs
        return self.execute_command(*args)
