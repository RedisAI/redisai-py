from redis import StrictRedis

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

    def __init__(self, ttype, shape, value):
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

    def __init__(self, ttype, shape, *blobs):
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
            blobs = blobarr
        else:
            blobs = blobs[0]
            size = 1

        super(BlobTensor, self).__init__(ttype, shape, blobs)
        self._size = size


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
        """
        Set the values of the tensor on the server using the provided Tensor object
        :param key: The name of the tensor
        :param tensor: a `Tensor` object
        :return:
        """
        args = ['AI.TENSORSET', key, tensor.type, tensor.size]
        args += tensor.shape
        args += [tensor.ARGNAME]
        args += tensor.value
        print args
        return self.execute_command(*args)

    def tensorget(self, key, astype=Tensor, meta_only=False):
        argname = 'META' if meta_only else astype.ARGNAME
        res = self.execute_command('AI.TENSORGET', key, argname)
        if meta_only:
            return astype(res[0], res[1], [])
        else:
            dtype, shape, value = res
            return astype(dtype, shape, value)

    def scriptset(self, name, device, script):
        return self.execute_command('AI.SCRIPTSET', name, device, script)

    def scriptget(self, name):
        r = self.execute_command('AI.SCRIPTGET', name)
        return {
            'device': r[0],
            'script': r[1]
        }

    def scriptrun(self, name, function, inputs, outputs):
        args = ['AI.SCRIPTRUN', name, function, 'INPUTS']
        args += inputs
        args += ['OUTPUTS']
        args += outputs
        return self.execute_command(*args)
