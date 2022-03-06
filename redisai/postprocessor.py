from typing import Any, Dict, List, Union, overload

import numpy as np

from . import utils


def decoder(val):
    return val.decode()


class Processor:
    @staticmethod
    def modelget(res):
        resdict = utils.list2dict(res)
        utils.recursive_bytetransform(resdict["inputs"], lambda x: x.decode())
        utils.recursive_bytetransform(resdict["outputs"], lambda x: x.decode())
        return resdict

    @staticmethod
    def modelscan(res):
        return utils.recursive_bytetransform(res, lambda x: x.decode())

    @overload
    @staticmethod
    def tensorget(res: List[Any], as_numpy: bool = True, as_numpy_mutable: bool = False, meta_only: bool = False) -> np.ndarray:
        ...

    @overload
    @staticmethod
    def tensorget(res: List[Any], as_numpy: bool = False, as_numpy_mutable: bool = True, meta_only: bool = False) -> np.ndarray:
        ...

    @overload
    @staticmethod
    def tensorget(res: List[Any], as_numpy: bool = False, as_numpy_mutable: bool = False, meta_only: bool = True) -> Dict[str, Any]:
        ...

    @overload
    @staticmethod
    def tensorget(res: List[Any]) -> List[Any]:
        ...

    @staticmethod
    def tensorget(res: List[Any], as_numpy: bool = False, as_numpy_mutable: bool = False, meta_only: bool = False) -> Any:
        """Process the tensorget output.

        If ``as_numpy`` is True, it'll be converted to a numpy array. The required
        information such as datatype and shape must be in ``rai_result`` itself.
        """
        if (as_numpy and as_numpy_mutable) or (as_numpy and meta_only) or (as_numpy_mutable and meta_only):
            raise Exception("Only one parameter should be set to true")
        rai_result = utils.list2dict(res)
        if meta_only is True:
            return rai_result
        elif as_numpy_mutable is True:
            return utils.blob2numpy(
                rai_result["blob"],
                rai_result["shape"],
                rai_result["dtype"],
                mutable=True,
            )
        elif as_numpy is True:
            return utils.blob2numpy(
                rai_result["blob"],
                rai_result["shape"],
                rai_result["dtype"],
                mutable=False,
            )
        else:
            if rai_result["dtype"] == "STRING":
                def target(b):
                    return b.decode()
            else:
                target = float if rai_result["dtype"] in ("FLOAT", "DOUBLE") else int
            utils.recursive_bytetransform(rai_result["values"], target)
            return rai_result

    @staticmethod
    def scriptget(res):
        return utils.list2dict(res)

    @staticmethod
    def scriptscan(res):
        return utils.recursive_bytetransform(res, lambda x: x.decode())

    @staticmethod
    def infoget(res):
        return utils.list2dict(res)


# These functions are only doing decoding on the output from redis
decoder = staticmethod(decoder)
decoding_functions = (
    "loadbackend",
    "modelstore",
    "modelset",
    "modeldel",
    "modelexecute",
    "modelrun",
    "tensorset",
    "scriptset",
    "scriptstore",
    "scriptdel",
    "scriptrun",
    "scriptexecute",
    "inforeset",
)
for fn in decoding_functions:
    setattr(Processor, fn, decoder)
