from typing import List, AnyStr, Callable

from redisai import utils


def decoder(val):
    return val.decode()


def _recursive_bytetransform(arr: List[AnyStr], target: Callable) -> list:
    """
    Recurse value, replacing each element of b'' with the appropriate element.
    Function returns the same array after inplace operation which updates `arr`
    """
    for ix in range(len(arr)):
        obj = arr[ix]
        if isinstance(obj, list):
            _recursive_bytetransform(obj, target)
        else:
            arr[ix] = target(obj)
    return arr


class Processor:
    @staticmethod
    def modelget(res):
        resdict = utils.list2dict(res)
        _recursive_bytetransform(resdict["inputs"], lambda x: x.decode())
        _recursive_bytetransform(resdict["outputs"], lambda x: x.decode())
        return resdict

    @staticmethod
    def modelscan(res):
        return _recursive_bytetransform(res, lambda x: x.decode())

    @staticmethod
    def tensorget(res, as_numpy, as_numpy_mutable, meta_only):
        """Process the tensorget output.

        If ``as_numpy`` is True, it'll be converted to a numpy array. The required
        information such as datatype and shape must be in ``rai_result`` itself.
        """
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
            target = float if rai_result["dtype"] in ("FLOAT", "DOUBLE") else int
            _recursive_bytetransform(rai_result["values"], target)
            return rai_result

    @staticmethod
    def scriptget(res):
        return utils.list2dict(res)

    @staticmethod
    def scriptscan(res):
        return _recursive_bytetransform(res, lambda x: x.decode())

    @staticmethod
    def infoget(res):
        return utils.list2dict(res)


# These functions are only doing decoding on the output from redis
decoder = staticmethod(decoder)
decoding_functions = (
    "loadbackend",
    "modelset",
    "modeldel",
    "modelrun",
    "tensorset",
    "scriptset",
    "scriptdel",
    "scriptrun",
    "inforeset",
)
for fn in decoding_functions:
    setattr(Processor, fn, decoder)
