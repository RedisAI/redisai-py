from redis import StrictRedis
from redis._compat import (long, nativestr)
from enum import Enum
import six

class Type(Enum):
    FLOAT=1
    DOUBLE=2
    INT8=3
    INT16=4
    INT32=5
    INT64=6
    UINT8=7
    UINT16=8
    

class Client(StrictRedis):

    def __init__(self, *args, **kwargs):
        """
        Create a new Client optional host and port

        If conn is not None, we employ an already existing redis connection
        """
        StrictRedis.__init__(self, *args, **kwargs)
        
                # Set the module commands' callbacks
        MODULE_CALLBACKS = {
                'AI.TENSORSET': lambda r: r and nativestr(r) == 'OK',
        }
        for k, v in six.iteritems(MODULE_CALLBACKS):
            self.set_response_callback(k, v)


    def tensorset(self, key, type, dimensions, tensor):
        args = ['AI.TENSORSET', key, type.name] + dimensions + ['VALUES'] + tensor
        
        return self.execute_command(*args)

 