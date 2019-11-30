from redisai import Client, DType, Device, Backend
import ml2rt

client = Client()
client.tensorset('x', [2, 3], dtype=DType.float)
t = client.tensorget('x')
print(t.value)

model = ml2rt.load_model('test/testdata/graph.pb')
client.tensorset('a', (2, 3), dtype=DType.float)
client.tensorset('b', (12, 10), dtype=DType.float)
client.modelset('m', Backend.tf,
                Device.cpu,
                inputs=['a', 'b'],
                outputs='mul',
                data=model)
client.modelrun('m', ['a', 'b'], ['mul'])
print(client.tensorget('mul').value)

# Try with a script
script = ml2rt.load_script('test/testdata/script.txt')
client.scriptset('ket', Device.cpu, script)
client.scriptrun('ket', 'bar', inputs=['a', 'b'], outputs='c')

