from functools import partial
from typing import Any, AnyStr, List, Sequence, Union

import numpy as np

from redisai import command_builder as builder
from redisai.postprocessor import Processor

processor = Processor()


class Dag:
    def __init__(self, load, persist, keys, timeout, executor, readonly=False, postprocess=True):
        self.result_processors = []
        self.enable_postprocess = True
        if readonly:
            if persist:
                raise RuntimeError(
                    "READONLY requests cannot write (duh!) and should not "
                    "have PERSISTing values"
                )
            self.commands = ["AI.DAGEXECUTE_RO"]
        else:
            self.commands = ["AI.DAGEXECUTE"]
        if load is not None:
            if not isinstance(load, (list, tuple)):
                self.commands += ["LOAD", 1, load]
            else:
                self.commands += ["LOAD", len(load), *load]
        if persist is not None:
            if not isinstance(persist, (list, tuple)):
                self.commands += ["PERSIST", 1, persist]
            else:
                self.commands += ["PERSIST", len(persist), *persist]
        if keys is not None:
            if not isinstance(keys, (list, tuple)):
                self.commands += ["KEYS", 1, keys]
            else:
                self.commands += ["KEYS", len(keys), *keys]
        if timeout is not None:
            self.commands += ["TIMEOUT", timeout]

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

    def modelexecute(
        self,
        key: AnyStr,
        inputs: Union[AnyStr, List[AnyStr]],
        outputs: Union[AnyStr, List[AnyStr]],
    ) -> Any:
        args = builder.modelexecute(key, inputs, outputs, None)
        self.commands.extend(args)
        self.commands.append("|>")
        self.result_processors.append(bytes.decode)
        return self

    def execute(self):
        commands = self.commands[:-1]  # removing the last "|>"
        results = self.executor(*commands)
        if self.enable_postprocess:
            out = []
            for res, fn in zip(results, self.result_processors):
                out.append(fn(res))
        else:
            out = results
        return out
