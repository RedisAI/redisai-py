from typing import Union, AnyStr, ByteString, List, Sequence
import numpy as np
from . import utils

# TODO: mypy check


class Builder:
    """ Command Builder class """
    @staticmethod
    def loadbackend(identifier: AnyStr, path: AnyStr) -> Sequence:
        return 'AI.CONFIG LOADBACKEND', identifier, path

    def modelset(self, name: AnyStr, backend: str, device: str, data: ByteString,
                 batch: int, minbatch: int, tag: AnyStr, inputs: List[AnyStr],
                 outputs: List[AnyStr]) -> Sequence:
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
            args += ['INPUTS'] + utils.listify(inputs)
            args += ['OUTPUTS'] + utils.listify(outputs)
        args.append(data)
        return args

    def modelget(self, name: AnyStr, meta_only=False) -> Sequence:
        args = ['AI.MODELGET', name, 'META']
        if not meta_only:
            args.append('BLOB')
        return args

    def modeldel(self, name: AnyStr) -> Sequence:
        return 'AI.MODELDEL', name

    def modelrun(self, name: AnyStr, inputs: List[AnyStr], outputs: List[AnyStr]) -> Sequence:
        args = ('AI.MODELRUN', name, 'INPUTS', *utils.listify(inputs), 'OUTPUTS',
                *utils.listify(outputs))
        return args

    def modelscan(self) -> Sequence:
        return ("AI._MODELSCAN",)

    def tensorset(self,
                  key: AnyStr,
                  tensor: Union[np.ndarray, list, tuple],
                  shape: Sequence[int] = None,
                  dtype: str = None) -> Sequence:
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
        return args

    def tensorget(self,
                  key: AnyStr, as_numpy: bool = True,
                  meta_only: bool = False) -> Sequence:
        args = ['AI.TENSORGET', key, 'META']
        if not meta_only:
            if as_numpy is True:
                args.append('BLOB')
            else:
                args.append('VALUES')
        return args

    def scriptset(self, name: AnyStr, device: str, script: str, tag: AnyStr = None) -> Sequence:
        args = ['AI.SCRIPTSET', name, device]
        if tag:
            args += ['TAG', tag]
        args.append(script)
        return args

    def scriptget(self, name: AnyStr, meta_only=False) -> dict:
        # TODO scripget test
        args = ['AI.SCRIPTGET', name, 'META']
        if not meta_only:
            args.append('SOURCE')
        return args

    def scriptdel(self, name: AnyStr) -> Sequence:
        return 'AI.SCRIPTDEL', name

    def scriptrun(self,
                  name: AnyStr,
                  function: AnyStr,
                  inputs: Union[AnyStr, Sequence[AnyStr]],
                  outputs: Union[AnyStr, Sequence[AnyStr]]
                  ) -> Sequence:
        args = ('AI.SCRIPTRUN', name, function, 'INPUTS', *utils.listify(inputs), 'OUTPUTS',
                *utils.listify(outputs))
        return args

    def scriptscan(self) -> Sequence:
        return ("AI._SCRIPTSCAN",)

    def infoget(self, key: AnyStr) -> Sequence:
        return 'AI.INFO', key

    def inforeset(self, key: AnyStr) -> Sequence:
        return 'AI.INFO', key, 'RESETSTAT'