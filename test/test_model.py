import time
import os

from unittest import TestCase
from redisai import model as raimodel
from redisai import Client, Backend, Device, Tensor, DType
import tensorflow as tf
import torch

def get_tf_graph():
	x = tf.placeholder(tf.float32, name='input')
	W = tf.Variable(5., name='W')
	b = tf.Variable(3., name='b')
	y = x * W + b
	y = tf.identity(y, name='output')


class MyModule(torch.jit.ScriptModule):
    def __init__(self):
        super(MyModule, self).__init__()

    @torch.jit.script_method
    def forward(self, a, b):
        return a + b


class ModelTestCase(TestCase):

	def get_client(self):
		return Client()

	def testTFGraph(self):
		y = get_tf_graph()
		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		path = f'{time.time()}.pb'
		raimodel.Model.save(sess, path, output=['output'])
		model = raimodel.Model.load(path)
		os.remove(path)
		con = self.get_client()
		con.modelset(
			'tfmodel', Backend.tf, Device.cpu, model,
			input=['input'], output=['output'])
		con.tensorset('a', Tensor.scalar(DType.float, 2))
		con.modelrun('tfmodel', ['a'], 'c')
		tensor = con.tensorget('c')
		self.assertEqual([13], tensor.value)

	def testPyTorchGraph(self):
		torch_graph = MyModule()
		path = f'{time.time()}.pb'
		raimodel.Model.save(torch_graph, path)
		model = raimodel.Model.load(path)
		os.remove(path)
		con = self.get_client()
		con.modelset('ptmodel', Backend.torch, Device.cpu, model)
		con.tensorset('a', Tensor.scalar(DType.float, 2, 5))
		con.tensorset('b', Tensor.scalar(DType.float, 3, 7))
		con.modelrun('ptmodel', ['a', 'b'], 'c')
		tensor = con.tensorget('c')
		self.assertEqual([5, 12], tensor.value)

	def testFakeObjSave(self):
		fakemodel = {}
		self.assertRaises(
			TypeError,
			raimodel.Model.save, fakemodel, 'fake.pt')
		wrongmodel_pt = torch.nn.Linear(2, 3)
		self.assertRaises(
			TypeError,
			raimodel.Model.save, wrongmodel_pt, 'wrong.pt')

	def testScriptLoad(self):
		con = self.get_client()
		dirname = os.path.dirname(__file__)
		path = f'{dirname}/testdata/script.txt'
		script = raimodel.Model.load(path)
		con.scriptset('script', Device.cpu, script)
		con.tensorset('a', Tensor.scalar(DType.float, 2, 5))
		con.tensorset('b', Tensor.scalar(DType.float, 3, 7))
		con.scriptrun('script', 'bar', ['a', 'b'], 'c')
		tensor = con.tensorget('c')
		self.assertEqual([5, 12], tensor.value)

	def testPyTorchDevice(self):
		pass
