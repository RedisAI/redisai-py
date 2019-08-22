[![license](https://img.shields.io/github/license/RedisAI/redisai-py.svg)](https://github.com/RedisAI/redisai-py)
[![PyPI version](https://badge.fury.io/py/redisai.svg)](https://badge.fury.io/py/redisai)
[![CircleCI](https://circleci.com/gh/RedisAI/redisai-py/tree/master.svg?style=svg)](https://circleci.com/gh/RedisAI/redisai-py/tree/master)
[![GitHub issues](https://img.shields.io/github/release/RedisAI/redisai-py.svg)](https://github.com/RedisAI/redisai-py/releases/latest)
[![Codecov](https://codecov.io/gh/RedisAI/redisai-py/branch/master/graph/badge.svg)](https://codecov.io/gh/RedisAI/redisai-py)


# RedisAI Python Client


## Installation

1. Install Redis 5.0 or above

2. [Install RedisAI](http://redisai.io)

3. Install the python client

```sh
$ pip install redisai
```

4. Install serialization-deserialization utility (optional)
```sh
$ pip install ml2rt
```

[RedisAI example repo](https://github.com/RedisAI/redisai-examples) shows few examples made using redisai-py under `python_client` section. Checkout [ml2rt](https://github.com/hhsecond/ml2rt) for convenient functions those might help in converting models (sparkml, sklearn, xgboost to ONNX), serializing models to disk, loading it back to redisai-py etc.

For a quick walk through, checkout this example

```python
from redisai import Client
import numpy as np
from redisai import Tensor, BlobTensor, DType, Device, Backend
import ml2rt

client = Client()
arr = np.array([2, 3])
# `tensorset` accepts `numpy` array or `redisai.Tensor` objects
client.tensorset('x', arr)
t = client.tensorget('x')
print(t.value)

model = ml2rt.load_model('test/testdata/graph.pb')
client.tensorset('a', Tensor.scalar(DType.float, 2, 3))
client.tensorset('b', Tensor.scalar(DType.float, 12, 10))
client.modelset('m', Backend.tf,
                Device.cpu,
                input=['a', 'b'],
                output='mul',
                data=model)
client.modelrun('m', ['a', 'b'], ['mul'])
print(client.tensorget('mul').value)

# Try with a script
script = ml2rt.load_script('test/testdata/script.txt')
client.scriptset('ket', Device.cpu, script)
client.scriptrun('ket', 'bar', input=['a', 'b'], output='c')

b1 = client.tensorget('c', as_type=BlobTensor)
b2 = client.tensorget('c', as_type=BlobTensor)

client.tensorset('d', BlobTensor(DType.float, b1.shape, b1, b2))

tnp = b1.to_numpy()
print(tnp)
```


