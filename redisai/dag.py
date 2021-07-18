from functools import partial
from typing import Any, AnyStr, List, Sequence, Union

import numpy as np

from redisai import command_builder as builder
from redisai.postprocessor import Processor
from deprecated import deprecated

processor = Processor()


class Dag:
    def __init__(self, load, persist, routing, timeout, executor, readonly=False, postprocess=True):
        self.result_processors = []
        self.enable_postprocess = True
        if load is None and persist is None and routing is None:
            raise RuntimeError(
                "AI.DAGEXECUTE and AI.DAGEXECUTE_RO commands must contain"
                "at least one out of LOAD, PERSIST, ROUTING parameters"
            )
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
        if routing is not None:
            self.commands += ["ROUTING", routing]
        if timeout is not None:
            self.commands += ["TIMEOUT", timeout]

        self.commands.append("|>")
        self.executor = executor
        self.readonly = readonly

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

    @deprecated(version="1.2.0", reason="Use modelexecute instead")
    def modelrun(
            self,
            key: AnyStr,
            inputs: Union[AnyStr, List[AnyStr]],
            outputs: Union[AnyStr, List[AnyStr]],
    ) -> Any:
        return self.modelexecute(key, inputs, outputs)

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

    def scriptexecute(
        self,
        key: AnyStr,
        function: str,
        keys: Union[AnyStr, Sequence[AnyStr]] = None,
        inputs: Union[AnyStr, Sequence[AnyStr]] = None,
        args: Union[AnyStr, Sequence[AnyStr]] = None,
        outputs: Union[AnyStr, List[AnyStr]] = None,
    ) -> Any:
        if self.readonly:
            raise RuntimeError(
                "AI.SCRIPTEXECUTE cannot be used in readonly mode"
            )
        args = builder.scriptexecute(key, function, keys, inputs, args, outputs, None)
        self.commands.extend(args)
        self.commands.append("|>")
        self.result_processors.append(bytes.decode)
        return self

    @deprecated(version="1.2.0", reason="Use execute instead")
    def run(self):
        return self.execute()

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
