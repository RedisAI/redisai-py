from functools import wraps, partial
from typing import Union, AnyStr, ByteString, List, Sequence, Any
import warnings

from redis import StrictRedis
import numpy as np

from . import utils
from .command_builder import Builder


builder = Builder()


def enable_debug(f):
    @wraps(f)
    def wrapper(*args):
        print(*args)
        return f(*args)
    return wrapper


class Dag:
    def __init__(self, load, persist, executor, readonly=False):
        self.result_processors = []
        if readonly:
            if persist:
                raise RuntimeError("READONLY requests cannot write (duh!) and should not "
                                   "have PERSISTing values")
            self.commands = ['AI.DAGRUN_RO']
        else:
            self.commands = ['AI.DAGRUN']
        if load:
            if not isinstance(load, (list, tuple)):
                self.commands += ["LOAD", 1, load]
            else:
                self.commands += ["LOAD", len(load), *load]
        if persist:
            if not isinstance(persist, (list, tuple)):
                self.commands += ["PERSIST", 1, persist, '|>']
            else:
                self.commands += ["PERSIST", len(persist), *persist, '|>']
        elif load:
            self.commands.append('|>')
        self.executor = executor

    def tensorset(self,
                  key: AnyStr,
                  tensor: Union[np.ndarray, list, tuple],
                  shape: Sequence[int] = None,
                  dtype: str = None) -> Any:
        args = builder.tensorset(key, tensor, shape, dtype)
        self.commands.extend(args)
        self.commands.append("|>")
        self.result_processors.append(bytes.decode)
        return self

    def tensorget(self,
                  key: AnyStr, as_numpy: bool = True,
                  meta_only: bool = False) -> Any:
        args = builder.tensorget(key, as_numpy, meta_only)
        self.commands.extend(args)
        self.commands.append("|>")
        self.result_processors.append(partial(utils.tensorget_postprocessor,
                                              as_numpy,
                                              meta_only))
        return self

    def modelrun(self,
                 name: AnyStr,
                 inputs: Union[AnyStr, List[AnyStr]],
                 outputs: Union[AnyStr, List[AnyStr]]) -> Any:
        args = builder.modelrun(name, inputs, outputs)
        self.commands.extend(args)
        self.commands.append("|>")
        self.result_processors.append(bytes.decode)
        return self

    def run(self):
        results = self.executor(*self.commands)
        out = []
        for res, fn in zip(results, self.result_processors):
            out.append(fn(res))
        return out


class Client(StrictRedis):
    """
    RedisAI client that can call Redis with RedisAI specific commands
    """
    def __init__(self, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if debug:
            self.execute_command = enable_debug(super().execute_command)

    def dag(self, load: Sequence = None, persist: Sequence = None,
            readonly: bool = False) -> Dag:
        """ Special function to return a dag object """
        return Dag(load, persist, self.execute_command, readonly)

    def loadbackend(self, identifier: AnyStr, path: AnyStr) -> str:
        """
        RedisAI by default won't load any backends. User can either explicitly
        load the backend by using this function or let RedisAI load the required
        backend from the default path on-demand.

        :param identifier: String representing which backend. Allowed values - TF, TORCH & ONNX
        :param path: Path to the shared object of the backend
        :return: byte string represents success or failure
        """
        args = builder.loadbackend(identifier, path)
        return self.execute_command(*args).decode()

    def modelset(self,
                 name: AnyStr,
                 backend: str,
                 device: str,
                 data: ByteString,
                 batch: int = None,
                 minbatch: int = None,
                 tag: AnyStr = None,
                 inputs: Union[AnyStr, List[AnyStr]] = None,
                 outputs: Union[AnyStr, List[AnyStr]] = None) -> str:
        """
        Set the model on provided key.
        :param name: str, Key name
        :param backend: str, Backend name. Allowed backends are TF, TORCH, TFLITE, ONNX
        :param device: str, Device name. Allowed devices are CPU and GPU
        :param data: bytes, Model graph read as bytestring
        :param batch: int, Number of batches for doing autobatching
        :param minbatch: int, Minimum number of samples required in a batch for model
            execution
        :param tag: str, Any string that will be saved in RedisAI as tags for the model
        :param inputs: list, List of strings that represents the input nodes in the graph.
            Required only Tensorflow graphs
        :param outputs: list, List of strings that represents the output nodes in the graph
            Required only for Tensorflow graphs

        :return:
        """
        args = builder.modelset(name, backend, device, data,
                                batch, minbatch, tag, inputs, outputs)
        return self.execute_command(*args).decode()

    def modelget(self, name: AnyStr, meta_only=False) -> dict:
        args = builder.modelget(name, meta_only)
        rv = self.execute_command(*args)
        return utils.list2dict(rv)

    def modeldel(self, name: AnyStr) -> str:
        args = builder.modeldel(name)
        return self.execute_command(*args).decode()

    def modelrun(self,
                 name: AnyStr,
                 inputs: Union[AnyStr, List[AnyStr]],
                 outputs: Union[AnyStr, List[AnyStr]]) -> str:
        args = builder.modelrun(name, inputs, outputs)
        return self.execute_command(*args).decode()

    def modelscan(self) -> list:
        warnings.warn("Experimental: Model List API is experimental and might change "
                      "in the future without any notice", UserWarning)
        args = builder.modelscan()
        result = self.execute_command(*args)
        return utils.recursive_bytetransform(result, lambda x: x.decode())

    def tensorset(self,
                  key: AnyStr,
                  tensor: Union[np.ndarray, list, tuple],
                  shape: Sequence[int] = None,
                  dtype: str = None) -> str:
        """
        Set the values of the tensor on the server using the provided Tensor object
        :param key: The name of the tensor
        :param tensor: a `np.ndarray` object or python list or tuple
        :param shape: Shape of the tensor. Required if `tensor` is list or tuple
        :param dtype: data type of the tensor. Required if `tensor` is list or tuple
        """
        args = builder.tensorset(key, tensor, shape, dtype)
        return self.execute_command(*args).decode()

    def tensorget(self,
                  key: AnyStr, as_numpy: bool = True,
                  meta_only: bool = False) -> Union[dict, np.ndarray]:
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
        args = builder.tensorget(key, as_numpy, meta_only)
        res = self.execute_command(*args)
        return utils.tensorget_postprocessor(as_numpy, meta_only, res)

    def scriptset(self, name: AnyStr, device: str, script: str, tag: AnyStr = None) -> str:
        args = builder.scriptset(name, device, script, tag)
        return self.execute_command(*args).decode()

    def scriptget(self, name: AnyStr, meta_only=False) -> dict:
        # TODO scripget test
        args = builder.scriptget(name, meta_only)
        ret = self.execute_command(*args)
        return utils.list2dict(ret)

    def scriptdel(self, name: AnyStr) -> str:
        args = builder.scriptdel(name)
        return self.execute_command(*args).decode()

    def scriptrun(self,
                  name: AnyStr,
                  function: AnyStr,
                  inputs: Union[AnyStr, Sequence[AnyStr]],
                  outputs: Union[AnyStr, Sequence[AnyStr]]
                  ) -> str:
        args = builder.scriptrun(name, function, inputs, outputs)
        out = self.execute_command(*args)
        return out.decode()

    def scriptscan(self) -> list:
        warnings.warn("Experimental: Script List API is experimental and might change "
                      "in the future without any notice", UserWarning)
        args = builder.scriptscan()
        return utils.recursive_bytetransform(self.execute_command(*args), lambda x: x.decode())

    def infoget(self, key: AnyStr) -> dict:
        args = builder.infoget(key)
        ret = self.execute_command(*args)
        return utils.list2dict(ret)

    def inforeset(self, key: AnyStr) -> str:
        args = builder.inforeset(key)
        return self.execute_command(*args).decode()
