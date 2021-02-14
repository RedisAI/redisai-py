from typing import AnyStr, Union, Sequence, Any, List

import numpy as np

from redisai import command_builder as builder
from redisai import utils


class Dag:
    def __init__(self, load, persist, executor, readonly=False):
        self.result_processors = []
        if readonly:
            if persist:
                raise RuntimeError(
                    "READONLY requests cannot write (duh!) and should not "
                    "have PERSISTing values"
                )
            self.commands = ["AI.DAGRUN_RO"]
        else:
            self.commands = ["AI.DAGRUN"]
        if load:
            if not isinstance(load, (list, tuple)):
                self.commands += ["LOAD", 1, load]
            else:
                self.commands += ["LOAD", len(load), *load]
        if persist:
            if not isinstance(persist, (list, tuple)):
                self.commands += ["PERSIST", 1, persist, "|>"]
            else:
                self.commands += ["PERSIST", len(persist), *persist, "|>"]
        else:
            self.commands.append("|>")
        self.executor = executor

    def tensorset(
        self,
        key: AnyStr,
        tensor: Union[np.ndarray, list, tuple],
        shape: Sequence[int] = None,
        dtype: str = None,
    ) -> Any:
        args = builder.tensorset(key, tensor, shape, dtype)
        self.commands.extend(args)
        self.commands.append("|>")
        self.result_processors.append(None)
        return self

    def tensorget(
        self, key: AnyStr, as_numpy: bool = True, meta_only: bool = False
    ) -> Any:
        args = builder.tensorget(key, as_numpy, meta_only)
        self.commands.extend(args)
        self.commands.append("|>")
        if meta_only and as_numpy:
            raise ValueError("`as_numpy` and `meta_only` cannot be True together")
        elif as_numpy:
            self.result_processors.append(utils.list2numpy)
        else:
            self.result_processors.append(None)
        return self

    def modelrun(
        self,
        key: AnyStr,
        inputs: Union[AnyStr, List[AnyStr]],
        outputs: Union[AnyStr, List[AnyStr]],
    ) -> Any:
        args = builder.modelrun(key, inputs, outputs)
        self.commands.extend(args)
        self.commands.append("|>")
        self.result_processors.append(None)
        return self

    def run(self):
        results = self.executor(*self.commands)
        out = []
        for res, processor in zip(results, self.result_processors):
            tmp = processor(res) if processor else res
            out.append(tmp)
        return out
