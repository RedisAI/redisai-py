from functools import partial
from typing import AnyStr, Union, Sequence, Any, List

import numpy as np

from redisai.postprocessor import Processor
from redisai import command_builder as builder


processor = Processor()


class Dag:
    def __init__(self, load, persist, executor, readonly=False, postprocess=True):
        self.result_processors = []
        self.enable_postprocess = True
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
        self.result_processors.append(bytes.decode)
        return self

    def tensorget(
        self,
        key: AnyStr,
        as_numpy: bool = True,
        as_numpy_mutable: bool = False,
        meta_only: bool = False,
    ) -> Any:
        args = builder.tensorget(key, as_numpy, as_numpy_mutable)
        self.commands.extend(args)
        self.commands.append("|>")
        self.result_processors.append(
            partial(
                processor.tensorget,
                as_numpy=as_numpy,
                as_numpy_mutable=as_numpy_mutable,
                meta_only=meta_only,
            )
        )
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
        self.result_processors.append(bytes.decode)
        return self

    def run(self):
        commands = self.commands[:-1]  # removing the last "|>
        results = self.executor(*commands)
        if self.enable_postprocess:
            out = []
            for res, fn in zip(results, self.result_processors):
                out.append(fn(res))
        else:
            out = results
        return out
