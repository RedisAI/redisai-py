from typing import Union, List, Sequence

import numpy as np

from redisai.redisai_ops import RedisAIOpsMixin
from redisai.dag import Dag
from redisai.pipeline import Pipeline


class Client(RedisAIOpsMixin):
    def pipeline(self, transaction: bool = True, shard_hint: bool = None) -> "Pipeline":
        """
        It follows the same pipeline implementation of native redis client but enables it
        to access redisai operation as well. This function is experimental in the
        current release.
        # TODO: Use the `transaction` and `shared_hint` args from user

        Example
        -------
        >>> pipe = con.pipeline(transaction=False)
        >>> pipe = pipe.set('nativeKey', 1)
        >>> pipe = pipe.tensorset('redisaiKey', np.array([1, 2]))
        >>> pipe.execute()
        [True, b'OK']
        """
        return Pipeline(
            self.connection_pool,
            self.response_callbacks,
            transaction=True,
            shard_hint=None,
        )

    def dag(
        self, load: Sequence = None, persist: Sequence = None, readonly: bool = False
    ) -> "Dag":
        """
        It returns a DAG object on which other DAG-allowed operations can be called. For
        more details about DAG in RedisAI, refer to the RedisAI documentation.

        Parameters
        ----------
        load : Union[AnyStr, List[AnyStr]]
            Load the list of given values from the keyspace to DAG scope
        persist : Union[AnyStr, List[AnyStr]]
            Write the list of given key, values to the keyspace from DAG scope
        readonly : bool
            If True, it triggers AI.DAGRUN_RO, the read only DAG which cannot write (PERSIST) to
            the keyspace. But since it can't write, it can execute on replicas


        Returns
        -------
        Any
            Dag object which holds other operations permitted inside DAG as attributes

        Example
        -------
        >>> con.tensorset('tensor', ...)
        'OK'
        >>> con.modelset('model', ...)
        'OK'
        >>> dag = con.dag(load=['tensor'], persist=['output'])
        >>> dag.tensorset('another', ...)
        >>> dag.modelrun('model', inputs=['tensor', 'another'], outputs=['output'])
        >>> output = dag.tensorget('output').run()
        >>> # You can even chain the operations
        >>> result = dag.tensorset(**akwargs).modelrun(**bkwargs).tensorget(**ckwargs).run()
        """
        return Dag(load, persist, self.execute_command, readonly)
