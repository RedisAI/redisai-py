import warnings

from redis.client import Pipeline as RedisPipeline
from redisai import utils
from redisai.base import BaseClient
from redisai import command_builder as builder


class Pipeline(RedisPipeline, BaseClient):
    def __init__(self, *args, **kwargs):
        warnings.warn("Pipeling AI commands through this client is experimental.",
                      UserWarning)
        self.index4numpy_convert = []
        super().__init__(*args, **kwargs)

    def dag(self, *args, **kwargs):
        raise RuntimeError("Pipeline object doesn't allow DAG creation")

    def tensorget(self, key, as_numpy=True, meta_only=False):
        if as_numpy and meta_only:
            raise ValueError("`as_numpy` and `meta_only` cannot be True together")
        elif as_numpy:
            self.index4numpy_convert.append(len(self.command_stack))
        args = builder.tensorget(key, as_numpy, meta_only)
        res = self.execute_command(*args)
        return res

    def _execute_transaction(self, *args, **kwargs):
        # TODO: Blocking commands like MODELRUN, SCRIPTRUN and DAGRUN won't work
        res = super()._execute_transaction(*args, **kwargs)
        # TODO: test this index4numpy thing
        for i in self.index4numpy_convert:
            res[i] = utils.list2numpy(res[i])
        return res

    def _execute_pipeline(self, *args, **kwargs):
        res = super()._execute_pipeline(*args, **kwargs)
        for i in self.index4numpy_convert:
            res[i] = utils.list2numpy(res[i])
        return res
