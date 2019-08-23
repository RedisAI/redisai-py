import numpy as np
from .utils import convert_to_num
from .constants import DType
try:
    from typing import Union, Any, AnyStr, ByteString, Collection, Type
except ImportError:
    pass


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
        convert_to_num(dtype, value)
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
        if isinstance(t, DType):
            t = t.value
        mm = {
            'FLOAT': 'float32',
            'DOUBLE': 'float64'
        }
        if t in mm:
            return mm[t]
        return t.lower()

    @classmethod
    def from_resp(cls, dtype, shape, value):
        return cls(dtype, shape, value)