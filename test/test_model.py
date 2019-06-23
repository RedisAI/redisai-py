import time
import os

from unittest import TestCase
from redisai import save_model, load_model
from redisai import Client, Backend, Device, Tensor, DType
import tensorflow as tf
import torch
from sklearn import linear_model, datasets
import onnx


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


def get_sklearn_model_and_prototype():
    model = linear_model.LinearRegression()
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    model.fit(X, y)
    return model, X[0].reshape(1, -1)


def get_onnx_model():
    torch_model = torch.nn.ReLU()
    # maybe there exists, but couldn't find a way to pass
    # the onnx model without writing to disk
    torch.onnx.export(torch_model, torch.rand(1, 1), 'model.onnx')
    onnx_model = onnx.load('model.onnx')
    os.remove('model.onnx')
    return onnx_model


class ModelTestCase(TestCase):

    def get_client(self):
        return Client()

    def testTFGraph(self):
        _ = get_tf_graph()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        path = f'{time.time()}.pb'
        save_model(sess, path, output=['output'])
        model = load_model(path)
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
        path = f'{time.time()}.pt'
        save_model(torch_graph, path)
        model = load_model(path)
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
            RuntimeError,
            save_model, fakemodel, 'fake.pt')
        wrongmodel_pt = torch.nn.Linear(2, 3)
        self.assertRaises(
            RuntimeError,
            save_model, wrongmodel_pt, 'wrong.pt')

    def testScriptLoad(self):
        con = self.get_client()
        dirname = os.path.dirname(__file__)
        path = f'{dirname}/testdata/script.txt'
        script = load_model(path)
        con.scriptset('script', Device.cpu, script)
        con.tensorset('a', Tensor.scalar(DType.float, 2, 5))
        con.tensorset('b', Tensor.scalar(DType.float, 3, 7))
        con.scriptrun('script', 'bar', ['a', 'b'], 'c')
        tensor = con.tensorget('c')
        self.assertEqual([5, 12], tensor.value)

    def testSKLearnGraph(self):
        sklearn_model, prototype = get_sklearn_model_and_prototype()
        path = f'{time.time()}.onnx'
        self.assertRaises(TypeError, save_model, sklearn_model, path)
        save_model(sklearn_model, path, prototype=prototype)
        model = load_model(path)
        os.remove(path)
        con = self.get_client()
        con.modelset('onnx_skl_model', Backend.onnx, Device.cpu, model)
        con.tensorset('a', Tensor.scalar(DType.float, *([1] * 13)))
        con.modelrun('onnx_skl_model', ['a'], ['outfromonnxskl'])
        tensor = con.tensorget('outfromonnxskl')
        self.assertEqual(len(tensor.value), 1)

    def testONNXGraph(self):
        onnx_model = get_onnx_model()
        path = f'{time.time()}.onnx'
        save_model(onnx_model, path)
        model = load_model(path)
        os.remove(path)
        con = self.get_client()
        con.modelset('onnxmodel', Backend.onnx, Device.cpu, model)
        con.tensorset('a', Tensor.scalar(DType.float, 2, -1))
        con.modelrun('onnxmodel', ['a'], ['c'])
        tensor = con.tensorget('c')
        self.assertEqual([2.0, 0.0], tensor.value)
