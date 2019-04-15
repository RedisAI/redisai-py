from __future__ import print_function
from redisai import Client, Tensor, ScalarTensor, \
    BlobTensor, DType, Device, Backend
from redisai import model as raimodel

client = Client()
client.tensorset('x', Tensor(DType.float, [2], [2, 3]))
t = client.tensorget('x')
print(t.value)

model = raimodel.Model.load('../RedisAI/examples/models/graph.pb')
client.tensorset('a', ScalarTensor(DType.float, 2, 3))
client.tensorset('b', ScalarTensor(DType.float, 12, 10))
client.modelset('m', Backend.tf,
                Device.cpu,
                input=['a', 'b'],
                output='mul',
                data=model)
client.modelrun('m', ['a', 'b'], ['mul'])
print(client.tensorget('mul').value)

# Try with a script
script = raimodel.Model.load('../RedisAI/examples/models/script.txt')
client.scriptset('ket', Device.cpu, script)
client.scriptrun('ket', 'bar', input=['a', 'b'], output='c')

b1 = client.tensorget('c', astype=BlobTensor)
b2 = client.tensorget('c', astype=BlobTensor)
bt = BlobTensor(DType.float, b1.shape, b1, b2)

print(len(bytes(bt.blob)))
print(bt.shape)

client.tensorset('d', BlobTensor(DType.float, b1.shape, b1, b2))

tnp = b1.to_numpy()
client.tensorset('e', tnp)