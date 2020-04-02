from redis import StrictRedis
from typing import Union, Any, AnyStr, ByteString, Sequence
from .containers import Script, Model, Tensor

try:
    import numpy as np
except ImportError:
    np = None

from .constants import Backend, Device, DType
from .utils import str_or_strsequence, to_string
from . import convert


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
                 inputs: Union[AnyStr, Sequence[AnyStr], None] = None,
                 outputs: Union[AnyStr, Sequence[AnyStr], None] = None
                 ) -> AnyStr:
        args = ['AI.MODELSET', name, backend.value, device.value]
        if backend == Backend.tf:
            if not(all((inputs, outputs))):
                raise ValueError(
                    'Require keyword arguments input and output for TF models')
            args += ['INPUTS'] + str_or_strsequence(inputs)
            args += ['OUTPUTS'] + str_or_strsequence(outputs)
        args += [data]
        return self.execute_command(*args)

    def modelget(self, name: AnyStr) -> Model:
        rv = self.execute_command('AI.MODELGET', name)
        return Model(
            rv[2],
            Device(to_string(rv[1])),
            Backend(to_string(rv[0])))

    def modeldel(self, name: AnyStr) -> AnyStr:
        return self.execute_command('AI.MODELDEL', name)

    def modelrun(self,
                 name: AnyStr,
                 inputs: Union[AnyStr, Sequence[AnyStr]],
                 outputs: Union[AnyStr, Sequence[AnyStr]]
                 ) -> AnyStr:
        args = ['AI.MODELRUN', name]
        args += ['INPUTS'] + str_or_strsequence(inputs)
        args += ['OUTPUTS'] + str_or_strsequence(outputs)
        return self.execute_command(*args)

    def tensorset(self,
                  key: AnyStr,
                  tensor: Union[np.ndarray, list, tuple],
                  shape: Union[Sequence[int], None] = None,
                  dtype: Union[DType, type, None] = None) -> Any:
        """
        Set the values of the tensor on the server using the provided Tensor object
        :param key: The name of the tensor
        :param tensor: a `np.ndarray` object or python list or tuple
        :param shape: Shape of the tensor. Required if `tensor` is list or tuple
        :param dtype: data type of the tensor. Required if `tensor` is list or tuple
        """
        if np and isinstance(tensor, np.ndarray):
            tensor = convert.from_numpy(tensor)
            args = ['AI.TENSORSET', key, tensor.dtype.value, *tensor.shape, tensor.argname, tensor.value]
        elif isinstance(tensor, (list, tuple)):
            if shape is None:
                shape = (len(tensor),)
            if not isinstance(dtype, DType):
                dtype = DType.__members__[np.dtype(dtype).name]
            tensor = convert.from_sequence(tensor, shape, dtype)
            args = ['AI.TENSORSET', key, tensor.dtype.value, *tensor.shape, tensor.argname, *tensor.value]
        return self.execute_command(*args)

    def tensorget(self,
                  key: AnyStr, as_numpy: bool = True,
                  meta_only: bool = False) -> Union[Tensor, np.ndarray]:
        """
        Retrieve the value of a tensor from the server. By default it returns the numpy array
        but it can be controlled using `as_type` argument and `meta_only` argument.
        :param key: the name of the tensor
        :param as_numpy: Should it return data as numpy.ndarray.
            Wraps with namedtuple if False. This flag also decides how to fetch the
            value from RedisAI server and could have performance implications
        :param meta_only: if true, then the value is not retrieved,
            only the shape and the type
        :return: an instance of as_type
        """
        if meta_only:
            argname = 'META'
        elif as_numpy is True:
            argname = 'BLOB'
        else:
            argname = 'VALUES'

        res = self.execute_command('AI.TENSORGET', key, argname)
        dtype, shape = to_string(res[0]), res[1]
        if meta_only:
            return convert.to_sequence([], shape, dtype)
        if as_numpy is True:
            return convert.to_numpy(res[2], shape, dtype)
        else:
            return convert.to_sequence(res[2], shape, dtype)

    def scriptset(self, name: AnyStr, device: Device, script: AnyStr) -> AnyStr:
        return self.execute_command('AI.SCRIPTSET', name, device.value, script)

    def scriptget(self, name: AnyStr) -> Script:
        r = self.execute_command('AI.SCRIPTGET', name)
        return Script(
            to_string(r[1]),
            Device(to_string(r[0])))

    def scriptdel(self, name):
        return self.execute_command('AI.SCRIPTDEL', name)

    def scriptrun(self,
                  name: AnyStr,
                  function: AnyStr,
                  inputs: Union[AnyStr, Sequence[AnyStr]],
                  outputs: Union[AnyStr, Sequence[AnyStr]]
                  ) -> AnyStr:
        args = ['AI.SCRIPTRUN', name, function, 'INPUTS']
        args += str_or_strsequence(inputs)
        args += ['OUTPUTS']
        args += str_or_strsequence(outputs)
        return self.execute_command(*args)
