from typing import Union, AnyStr, ByteString, List, Sequence
import numpy as np
from . import utils

# TODO: mypy check


def loadbackend(identifier: AnyStr, path: AnyStr) -> Sequence:
    return "AI.CONFIG LOADBACKEND", identifier, path


def modelset(
    name: AnyStr,
    backend: str,
    device: str,
    data: ByteString,
    batch: int,
    minbatch: int,
    tag: AnyStr,
    inputs: Union[AnyStr, List[AnyStr]],
    outputs: Union[AnyStr, List[AnyStr]],
) -> Sequence:
    if device.upper() not in utils.allowed_devices:
        raise ValueError(f"Device not allowed. Use any from {utils.allowed_devices}")
    if backend.upper() not in utils.allowed_backends:
        raise ValueError(f"Backend not allowed. Use any from {utils.allowed_backends}")
    args = ["AI.MODELSET", name, backend, device]

    if batch is not None:
        args += ["BATCHSIZE", batch]
    if minbatch is not None:
        args += ["MINBATCHSIZE", minbatch]
    if tag is not None:
        args += ["TAG", tag]

    if backend.upper() == "TF":
        if not (all((inputs, outputs))):
            raise ValueError("Require keyword arguments input and output for TF models")
        args += ["INPUTS", *utils.listify(inputs)]
        args += ["OUTPUTS", *utils.listify(outputs)]
    chunk_size = 500 * 1024 * 1024
    data_chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    # TODO: need a test case for this
    args += ["BLOB", *data_chunks]
    return args


def modelget(name: AnyStr, meta_only=False) -> Sequence:
    args = ["AI.MODELGET", name, "META"]
    if not meta_only:
        args.append("BLOB")
    return args


def modeldel(name: AnyStr) -> Sequence:
    return "AI.MODELDEL", name


def modelrun(name: AnyStr, inputs: List[AnyStr], outputs: List[AnyStr]) -> Sequence:
    args = (
        "AI.MODELRUN",
        name,
        "INPUTS",
        *utils.listify(inputs),
        "OUTPUTS",
        *utils.listify(outputs),
    )
    return args


def modelscan() -> Sequence:
    return ("AI._MODELSCAN",)


def tensorset(
    key: AnyStr,
    tensor: Union[np.ndarray, list, tuple],
    shape: Sequence[int] = None,
    dtype: str = None,
) -> Sequence:
    if np and isinstance(tensor, np.ndarray):
        dtype, shape, blob = utils.numpy2blob(tensor)
        args = ["AI.TENSORSET", key, dtype, *shape, "BLOB", blob]
    elif isinstance(tensor, (list, tuple)):
        try:
            dtype = utils.dtype_dict[dtype.lower()]
        except KeyError:
            raise TypeError(
                f"``{dtype}`` is not supported by RedisAI. Currently "
                f"supported types are {list(utils.dtype_dict.keys())}"
            )
        except AttributeError:
            raise TypeError(
                "tensorset() missing argument 'dtype' or value of 'dtype' is None"
            )
        if shape is None:
            shape = (len(tensor),)
        args = ["AI.TENSORSET", key, dtype, *shape, "VALUES", *tensor]
    else:
        raise TypeError(
            f"``tensor`` argument must be a numpy array or a list or a "
            f"tuple, but got {type(tensor)}"
        )
    return args


def tensorget(key: AnyStr, as_numpy: bool = True, meta_only: bool = False) -> Sequence:
    args = ["AI.TENSORGET", key, "META"]
    if not meta_only:
        if as_numpy is True:
            args.append("BLOB")
        else:
            args.append("VALUES")
    return args


def scriptset(name: AnyStr, device: str, script: str, tag: AnyStr = None) -> Sequence:
    if device.upper() not in utils.allowed_devices:
        raise ValueError(f"Device not allowed. Use any from {utils.allowed_devices}")
    args = ["AI.SCRIPTSET", name, device]
    if tag:
        args += ["TAG", tag]
    args.append("SOURCE")
    args.append(script)
    return args


def scriptget(name: AnyStr, meta_only=False) -> Sequence:
    args = ["AI.SCRIPTGET", name, "META"]
    if not meta_only:
        args.append("SOURCE")
    return args


def scriptdel(name: AnyStr) -> Sequence:
    return "AI.SCRIPTDEL", name


def scriptrun(
    name: AnyStr,
    function: AnyStr,
    inputs: Union[AnyStr, Sequence[AnyStr]],
    outputs: Union[AnyStr, Sequence[AnyStr]],
) -> Sequence:
    args = (
        "AI.SCRIPTRUN",
        name,
        function,
        "INPUTS",
        *utils.listify(inputs),
        "OUTPUTS",
        *utils.listify(outputs),
    )
    return args


def scriptscan() -> Sequence:
    return ("AI._SCRIPTSCAN",)


def infoget(key: AnyStr) -> Sequence:
    return "AI.INFO", key


def inforeset(key: AnyStr) -> Sequence:
    return "AI.INFO", key, "RESETSTAT"
