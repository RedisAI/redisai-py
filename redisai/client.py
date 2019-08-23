from redis import StrictRedis
from typing import Union, Any, AnyStr, ByteString, Collection, Type

try:
    import numpy as np
except ImportError:
    np = None

from .constants import Backend, Device, DType
from .utils import str_or_strlist, to_string
from .tensor import Tensor, BlobTensor


class Client(StrictRedis):
    """
    RedisAI client that can call Redis with RedisAI specific commands
    """
    def loadbackend(self, identifier: AnyStr, path: AnyStr) -> AnyStr:
        """
        RedisAI by default won't load any backends. User can either explicitly
        load the backend by using this function or let RedisAI load the required
        backend from the default path on-demand.

        :param identifier: String representing which backend. Allowed values - TF, TORCH & ONNX
        :param path: Path to the shared object of the backend
        :return: byte string represents success or failure
        """
        return self.execute_command('AI.CONFIG LOADBACKEND', identifier, path)

    def modelset(self,
                 name: AnyStr,
                 backend: Backend,
                 device: Device,
                 data: ByteString,
                 input: Union[AnyStr, Collection[AnyStr], None] = None,
                 output: Union[AnyStr, Collection[AnyStr], None] = None
                 ) -> AnyStr:
        args = ['AI.MODELSET', name, backend.value, device.value]
        if backend == Backend.tf:
            if not(all((input, output))):
                raise ValueError(
                    'Require keyword arguments input and output for TF models')
            args += ['INPUTS'] + str_or_strlist(input)
            args += ['OUTPUTS'] + str_or_strlist(output)
        args += [data]
        return self.execute_command(*args)

    def modelget(self, name: AnyStr) -> dict:
        rv = self.execute_command('AI.MODELGET', name)
        return {
            'backend': Backend(rv[0]),
            'device': Device(rv[1]),
            'data': rv[2]
        }

    def modeldel(self, name: AnyStr) -> AnyStr:
        return self.execute_command('AI.MODELDEL', name)

    def modelrun(self,
                 name: AnyStr,
                 input: Union[AnyStr, Collection[AnyStr]],
                 output: Union[AnyStr, Collection[AnyStr]]
                 ) -> AnyStr:
        args = ['AI.MODELRUN', name]
        args += ['INPUTS'] + str_or_strlist(input)
        args += ['OUTPUTS'] + str_or_strlist(output)
        return self.execute_command(*args)

    def tensorset(self,
                  key: AnyStr,
                  tensor: Union[Tensor, np.ndarray, list, tuple],
                  shape: Union[Collection[int], None] = None,
                  dtype: Union[DType, None] = None) -> Any:
        """
        Set the values of the tensor on the server using the provided Tensor object
        :param key: The name of the tensor
        :param tensor: a `Tensor` object
        :param shape: Shape of the tensor. Required if input is a sequence of ints/floats
        :param dtype: data type of the tensor. Required if input is a sequence of ints/floats
        """
        # TODO: tensorset will not accept BlobTensor or Tensor object in the future.
        # Keeping it in the current version for compatibility with the example repo
        if np and isinstance(tensor, np.ndarray):
            tensor = BlobTensor.from_numpy(tensor)
        elif hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
            raise TypeError('Numpy is not installed but the input tensor seem to be a numpy array')
        elif isinstance(tensor, (list, tuple)):
            tensor = Tensor(dtype, shape, tensor)
        args = ['AI.TENSORSET', key, tensor.type.value]
        args += tensor.shape
        args += [tensor.ARGNAME]
        args += tensor.value
        return self.execute_command(*args)

    def tensorget(self,
                  key: AnyStr, as_type: Type[Tensor] = None,
                  meta_only: bool = False) -> Union[Tensor, BlobTensor]:
        """
        Retrieve the value of a tensor from the server. By default it returns the numpy array
        but it can be controlled using `as_type` argument and `meta_only` argument.
        :param key: the name of the tensor
        :param as_type: the resultant tensor type. Returns numpy array if None
        :param meta_only: if true, then the value is not retrieved,
            only the shape and the type
        :return: an instance of as_type
        """
        # TODO; We might remove Tensor & BlobTensor in the future and `tensorget` will return
        # python list or numpy arrays or a namedtuple
        if meta_only:
            argname = 'META'
        elif as_type is None:
            argname = BlobTensor.ARGNAME
        else:
            argname = as_type.ARGNAME

        res = self.execute_command('AI.TENSORGET', key, argname)
        dtype, shape = to_string(res[0]), res[1]
        dt = DType.__members__[dtype.lower()]
        if meta_only:
            return Tensor(dt, shape, [])
        elif as_type is None:
            return BlobTensor.from_resp(dt, shape, res[2]).to_numpy()
        else:
            return as_type.from_resp(dt, shape, res[2])

    def scriptset(self, name: AnyStr, device: Device, script: AnyStr) -> AnyStr:
        return self.execute_command('AI.SCRIPTSET', name, device.value, script)

    def scriptget(self, name: AnyStr) -> dict:
        r = self.execute_command('AI.SCRIPTGET', name)
        return {
            'device': to_string(r[0]),
            'script': to_string(r[1])
        }

    def scriptdel(self, name):
        return self.execute_command('AI.SCRIPTDEL', name)

    def scriptrun(self,
                  name: AnyStr,
                  function: AnyStr,
                  input: Union[AnyStr, Collection[AnyStr]],
                  output: Union[AnyStr, Collection[AnyStr]]
                  ) -> AnyStr:
        args = ['AI.SCRIPTRUN', name, function, 'INPUTS']
        args += str_or_strlist(input)
        args += ['OUTPUTS']
        args += str_or_strlist(output)
        return self.execute_command(*args)
