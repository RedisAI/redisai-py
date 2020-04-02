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


## Documentation
APIs available in redisai-py

### tensorset
Set the values of the tensor on the server using the provided tensor object

##### Parameters
- name: Key on which tensor is saved
- tensor: a `Tensor` object
- shape: Shape of the tensor
- dtype: redisai.DType object represents data type of the tensor. Required if input is a `list`/`tuple`

##### Example
```python
from redisai import Client, DType
client = Client()
arr = np.array([2, 3])
client.tensorset('x', arr)
client.tensorset('y', [1, 2], dtype=DType.float)
client.tensorset('z', [3, 4, 5, 6], dtype=DType.float, shape=(1, 2, 2))
```

### tensorget

##### Parameters
Retrieve the value of a tensor from the server. By default it returns the numpy array
but it can be controlled using `as_type` and `meta_only` arguments
- name: Key from where the tensor is saved
- as_type: the resultant tensor type. Returns numpy array if None
- meta_only: if true, then the value is not retrieved, only the shape and the type

##### Example
```python
from redisai import Tensor
x = client.tensorget('x')  # numpy array
y = client.tensorget('y', as_type=Tensor)  # A Tensor object
z = client.tensorget('z', meta_only=True)  # A Tensor object but without value
```

### loadbackend
RedisAI by default won't load any backends. User can either explicitly
load the backend by using this function or let RedisAI load the required
backend from the default path on-demand.

##### Parameters
- identifier: String representing which backend. Allowed values - TF, TORCH & ONNX
- path: Path to the shared object of the backend

##### Example
```python
client.loadbackend('TORCH', 'install-cpu/backends/redisai_torch/redisai_torch.so')
```


### modelset
Store a model of Tensorflow/PyTorch/ONNX format in RedisAI

##### Parameters
- name: Key on which model should be saved
- backend: redisai.Backend object - tf, torch or onnx
- device: redisai.Device object - cpu or gpu
- data: model as a byte string. `ml2rt.load_model(path)` returns this. 

##### Example
Tensorflow requires the input and output nodes of the graph while storing the model. For
exporting a normal tensorflow session to pb file, you could use `ml2rt` package
```python
import ml2rt
ml2rt.save_tensorflow(sess, 'path/to/graph.pb', output_nodes)
model = ml2rt.load_model('path/to/graph.pb')
client.modelset('m', Backend.tf,
                Device.cpu,
                input=['input_1', 'input_2'],
                output='output',
                data=model)
```
Torch doesn't need input and output node information. You could use ml2rt for exporting torch
model as well but ml2rt needs torchscript model rather than normal torch model. Checkout
the [document](https://pytorch.org/docs/stable/jit.html) to learn more. 
```python
import ml2rt
ml2rt.save_torch('optimized_graph', 'path/to/graph.pt')
model = ml2rt.load_model('path/to/graph.pt')
client.modelset('m', Backend.torch, Device.cpu, data=model)
```

### modelget
Fetch the stored model from RedisAI

##### Parameters
name: the name of the model

##### Example
```python
mod_det = client.modelget('m')
print(mod_det['backend'], mod_det['device'])
model_binary = mod_det['data']
```

### modeldel
Delete a stored model from RedisAI

##### Parameters
- name: Key of the model

##### Example
```python
client.modeldel('m')
```


### modelrun
Execute a model. Required inputs must be present in RedisAI before calling `modelrun`

##### Parameters
- name: Key of the model
- inputs: Key of the input tensors. It can be a single key or a list of keys
- outputs: Key on which the output tensors will be saved. It can be a single key or list of keys

##### Example
```python
client.tensorset('a', [2, 3], dtype=DType.float, shape=(2,))
client.tensorset('b', [12, 10], dtype=DType.float)
model = ml2rt.load_model('test/testdata/graph.pt')
client.modelset('m', Backend.torch,
                Device.cpu,
                input=['input_1', 'input_2'],
                output='output',
                data=model)
client.modelrun('m', ['a', 'b'], ['mul'])
out = client.tensorget('mul')
```

### scriptset
Store a SCRIPT in RedisAI. SCRIPT is a subset of python language itself but will be executed
on high performance C++ runtime. RedisAI uses TORCH runtime to execute SCRIPT and it must
follow the format required by the [doc](https://pytorch.org/docs/stable/jit.html).

##### Parameters
- name: Key on which SCRIPT should be saved
- device: redisai.Device object - cpu or gpu
- script: SCRIPT as defined in [TorchScript documentation](https://pytorch.org/docs/stable/jit.html). SCRIPT must have functions defined in it (you can have multiple functions).

##### Example
```python
script = """
def myfunc(a, b):
    return a + b
"""
client.scriptset('script', Device.cpu, script)
```


### scriptget
Fetch a stored SCRIPT from RedisAI

##### Parameters
- name: Key from which SCRIPT can be retrieved

##### Example
```python
script_details = client.scriptget('script')
device = script_details['device']
script = script_details['script']
```


### scriptdel
Delete a stored SCRIPT from RedisAI

##### Parameters
- name: Key from which SCRIPT can be retrieved

##### Example
```python
client.scriptdel('script')
```


### scriptrun
Execute a SCRIPT. Required inputs must be present in RedisAI before calling `modelrun`

##### Parameters
- name: Key from which SCRIPT can be retrieved
- function: name of the function to call. The function that you call can call other functions in the same SCRIPT
- inputs: Key of the input tensors. It can be a single key or a list of keys
- outputs: Key on which the output tensors will be saved. It can be a single key or list of keys

##### Example
```python
script = """
def myfunc(a, b):
    return a + b
"""
client.scriptset('script', Device.cpu, script)
client.tensorget('a', [1, 2], dtype=DType.float)
client.tensorget('b', [3, 4], dtype=DType.float)
client.scriptrun('script', 'myfunc', ['a', 'b'], 'out')
out = client.tensorget('out')  # => [4, 6]
```

