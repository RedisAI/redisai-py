from redisai import Client, Tensor, \
    BlobTensor, DType, Device, Backend
import mlut

client = Client()
client.tensorset('x', Tensor(DType.float, [2], [2, 3]))
t = client.tensorget('x')
print(t.value)

model = mlut.load_model('test/testdata/graph.pb')
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
script = mlut.load_script('test/testdata/script.txt')
client.scriptset('ket', Device.cpu, script)
client.scriptrun('ket', 'bar', input=['a', 'b'], output='c')

b1 = client.tensorget('c', as_type=BlobTensor)
b2 = client.tensorget('c', as_type=BlobTensor)

client.tensorset('d', BlobTensor(DType.float, b1.shape, b1, b2))

tnp = b1.to_numpy()
client.tensorset('e', tnp)
