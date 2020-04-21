from redis import StrictRedis
from typing import Union, AnyStr, ByteString, List, Sequence
import warnings
import numpy as np

from . import utils


class Client(StrictRedis):
    """
    RedisAI client that can call Redis with RedisAI specific commands
    """
    def loadbackend(self, identifier: AnyStr, path: AnyStr) -> str:
        """
        RedisAI by default won't load any backends. User can either explicitly
        load the backend by using this function or let RedisAI load the required
        backend from the default path on-demand.

        :param identifier: String representing which backend. Allowed values - TF, TORCH & ONNX
        :param path: Path to the shared object of the backend
        :return: byte string represents success or failure
        """
        return self.execute_command('AI.CONFIG LOADBACKEND', identifier, path).decode()

    def modelset(self,
                 name: AnyStr,
                 backend: str,
                 device: str,
                 data: ByteString,
                 batch: int = None,
                 minbatch: int = None,
                 tag: str = None,
                 inputs: List[AnyStr] = None,
                 outputs: List[AnyStr] = None) -> str:
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
        args = ['AI.MODELSET', name, backend, device]

        if batch is not None:
            args += ['BATCHSIZE', batch]
        if minbatch is not None:
            args += ['MINBATCHSIZE', minbatch]
        if tag is not None:
            args += ['TAG', tag]

        if backend.upper() == 'TF':
            if not(all((inputs, outputs))):
                raise ValueError(
                    'Require keyword arguments input and output for TF models')
            args += ['INPUTS'] + inputs
            args += ['OUTPUTS'] + outputs
        args.append(data)
        return self.execute_command(*args).decode()

    def modelget(self, name: AnyStr, meta_only=False) -> dict:
        argname = 'META' if meta_only else 'BLOB'
        rv = self.execute_command('AI.MODELGET', name, argname)
        return utils.list2dict(rv)

    def modeldel(self, name: AnyStr) -> str:
        return self.execute_command('AI.MODELDEL', name).decode()

    def modelrun(self,
                 name: AnyStr,
                 inputs: List[AnyStr],
                 outputs: List[AnyStr]
                 ) -> str:
        args = ['AI.MODELRUN', name, 'INPUTS', *inputs, 'OUTPUTS', *outputs]
        return self.execute_command(*args).decode()

    def modelist(self):
        # TODO: typing and consistent return
        warnings.warn("Experimental: Model List API is experimental and might change "
                      "in the future without any notice", UserWarning)
        return utils.un_bytize(self.execute_command("AI._MODELLIST"), lambda x: x.decode())

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
        if np and isinstance(tensor, np.ndarray):
            dtype, shape, blob = utils.numpy2blob(tensor)
            args = ['AI.TENSORSET', key, dtype, *shape, 'BLOB', blob]
        elif isinstance(tensor, (list, tuple)):
            if shape is None:
                shape = (len(tensor),)
            args = ['AI.TENSORSET', key, dtype, *shape, 'VALUES', *tensor]
        else:
            raise TypeError(f"``tensor`` argument must be a numpy array or a list or a "
                            f"tuple, but got {type(tensor)}")
        return self.execute_command(*args).decode()

    def tensorget(self,
                  key: str, as_numpy: bool = True,
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
        if meta_only:
            argname = 'META'
        elif as_numpy is True:
            argname = 'BLOB'
        else:
            argname = 'VALUES'

        res = self.execute_command('AI.TENSORGET', key, argname)
        dtype, shape = res[0].decode(), res[1]
        if meta_only:
            return {'shape': shape, 'dtype': dtype}
        elif as_numpy is True:
            return utils.blob2numpy(res[2], shape, dtype)
        else:
            target = float if dtype in ('FLOAT', 'DOUBLE') else int
            values = utils.un_bytize(res[2], target)
            return {'values': values, 'shape': shape, 'dtype': dtype}

    def scriptset(self, name: str, device: str, script: str, tag: str = None) -> str:
        args = ['AI.SCRIPTSET', name, device]
        if tag:
            args += ['TAG', tag]
        args.append(script)
        return self.execute_command(*args).decode()

    def scriptget(self, name: AnyStr) -> dict:
        ret = self.execute_command('AI.SCRIPTGET', name)
        return utils.list2dict(ret)

    def scriptdel(self, name: str) -> str:
        return self.execute_command('AI.SCRIPTDEL', name).decode()

    def scriptrun(self,
                  name: AnyStr,
                  function: AnyStr,
                  inputs: Sequence[AnyStr],
                  outputs: Sequence[AnyStr]
                  ) -> AnyStr:
        args = ['AI.SCRIPTRUN', name, function, 'INPUTS', *inputs, 'OUTPUTS', *outputs]
        return self.execute_command(*args).decode()

    def scriptlist(self):
        # TODO: Typing and consisent return
        warnings.warn("Experimental: Script List API is experimental and might change "
                      "in the future without any notice", UserWarning)
        return utils.un_bytize(self.execute_command("AI._SCRIPTLIST"), lambda x: x.decode())

    def infoget(self, key: str) -> dict:
        ret = self.execute_command('AI.INFO', key)
        return utils.list2dict(ret)

    def inforeset(self, key: str) -> str:
        return self.execute_command('AI.INFO', key, 'RESETSTAT').decode()
