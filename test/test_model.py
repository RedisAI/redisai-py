import time
import os
import sys
from unittest import TestCase

from redisai import save_model, load_model
from redisai import (
    save_tensorflow, save_torch, save_onnx, save_sklearn, save_sparkml)
from redisai.model import onnx_utils
from redisai import Client, Backend, Device, Tensor, DType, BlobTensor
import tensorflow as tf
import torch
from sklearn import linear_model, datasets
import onnx
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
import pyspark


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
    return model, X[0].reshape(1, -1).astype(np.float32)


def get_dummy_prototype():
    return np.array([1, 2])


def get_onnx_model():
    torch_model = torch.nn.ReLU()
    # maybe there exists, but couldn't find a way to pass
    # the onnx model without writing to disk
    torch.onnx.export(torch_model, torch.rand(1, 1), 'model.onnx')
    onnx_model = onnx.load('model.onnx')
    os.remove('model.onnx')
    return onnx_model


def get_spark_model_and_prototype():
    executable = sys.executable
    os.environ["SPARK_HOME"] = pyspark.__path__[0]
    os.environ["PYSPARK_PYTHON"] = executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = executable
    spark = SparkSession.builder.appName("redisai_test").getOrCreate()
    # label is input + 1
    data = spark.createDataFrame([
        (2.0, Vectors.dense(1.0)),
        (3.0, Vectors.dense(2.0)),
        (4.0, Vectors.dense(3.0)),
        (5.0, Vectors.dense(4.0)),
        (6.0, Vectors.dense(5.0)),
        (7.0, Vectors.dense(6.0))
    ], ["label", "features"])
    lr = LinearRegression(maxIter=5, regParam=0.0, solver="normal")
    model = lr.fit(data)
    prototype = np.array([[1.0]], dtype=np.float32)
    return model, prototype


class ModelTestCase(TestCase):

    def get_client(self):
        return Client()

    def testTFGraph(self):
        _ = get_tf_graph()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        path = f'{time.time()}.pb'
        save_tensorflow(sess, path, output=['output'])
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
        save_torch(torch_graph, path)
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
        prototype = get_dummy_prototype()
        self.assertRaises(
            AttributeError,
            save_torch, fakemodel, 'fake.pt')
        wrongmodel_pt = torch.nn.Linear(2, 3)
        self.assertRaises(
            AttributeError,
            save_torch, wrongmodel_pt, 'wrong.pt')
        self.assertRaises(
            AttributeError,
            save_tensorflow, wrongmodel_pt, 'wrong.pt', output=['output'])
        self.assertRaises(
            AttributeError,
            save_onnx, wrongmodel_pt, 'wrong.pt')
        self.assertRaises(
            RuntimeError,
            save_sklearn, wrongmodel_pt, 'wrong.pt', prototype=prototype)
        if os.path.isfile('wrong.pt'):
            os.remove('wrong.pt')

    def testSaveModelDeprecation(self):
        torch_graph = MyModule()
        self.assertRaises(
            DeprecationWarning,
            save_model, torch_graph, 'wrong.pt')

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

    def testONNXGraph(self):
        onnx_model = get_onnx_model()
        path = f'{time.time()}.onnx'
        save_onnx(onnx_model, path)
        model = load_model(path)
        os.remove(path)
        con = self.get_client()
        con.modelset('onnxmodel', Backend.onnx, Device.cpu, model)
        con.tensorset('a', Tensor.scalar(DType.float, 2, -1))
        con.modelrun('onnxmodel', ['a'], ['c'])
        tensor = con.tensorget('c')
        self.assertEqual([2.0, 0.0], tensor.value)

    def testSKLearnGraph(self):
        sklearn_model, prototype = get_sklearn_model_and_prototype()
        path = f'{time.time()}.onnx'
        self.assertRaises(RuntimeError, save_sklearn, sklearn_model, path)
        con = self.get_client()

        # saving with prototype
        path = f'{time.time()}.onnx'
        save_sklearn(sklearn_model, path, prototype=prototype)
        model = load_model(path)
        os.remove(path)
        con.modelset('onnx_skl_model1', Backend.onnx, Device.cpu, model)
        con.tensorset('a1', Tensor.scalar(DType.float, *([1] * 13)))
        con.modelrun('onnx_skl_model1', ['a1'], ['outfromonnxskl1'])
        tensor = con.tensorget('outfromonnxskl1')
        self.assertEqual(len(tensor.value), 1)

        # saving with shape and dtype
        shape = prototype.shape
        if prototype.dtype == np.float32:
            dtype = DType.float32
        else:
            raise RuntimeError("Test is not configured to run with another type")
        path = f'{time.time()}.onnx'
        save_sklearn(sklearn_model, path, shape=shape, dtype=dtype)
        model = load_model(path)
        os.remove(path)
        con.modelset('onnx_skl_model2', Backend.onnx, Device.cpu, model)
        con.tensorset('a2', Tensor.scalar(DType.float, *([1] * 13)))
        con.modelrun('onnx_skl_model2', ['a2'], ['outfromonnxskl2'])
        tensor = con.tensorget('outfromonnxskl2')
        self.assertEqual(len(tensor.value), 1)

        # saving with initial_types
        inital_types = onnx_utils.get_tensortype(shape, dtype)
        path = f'{time.time()}.onnx'
        save_sklearn(sklearn_model, path, initial_types=[inital_types])
        model = load_model(path)
        os.remove(path)
        con.modelset('onnx_skl_model3', Backend.onnx, Device.cpu, model)
        con.tensorset('a3', Tensor.scalar(DType.float, *([1] * 13)))
        con.modelrun('onnx_skl_model3', ['a3'], ['outfromonnxskl3'])
        tensor = con.tensorget('outfromonnxskl3')
        self.assertEqual(len(tensor.value), 1)

    def testSparkMLGraph(self):
        spark_model, prototype = get_spark_model_and_prototype()
        in_tensor = BlobTensor.from_numpy(prototype)
        path = f'{time.time()}.onnx'
        self.assertRaises(RuntimeError, save_sparkml, spark_model, path)
        con = self.get_client()

        # saving with prototype
        path = f'{time.time()}.onnx'
        save_sparkml(spark_model, path, prototype=prototype)
        model = load_model(path)
        os.remove(path)
        con.modelset('spark_model1', Backend.onnx, Device.cpu, model)
        con.tensorset('a1', in_tensor)
        con.modelrun('spark_model1', ['a1'], ['outfromspark1'])
        tensor = con.tensorget('outfromspark1')
        self.assertEqual(len(tensor.value), 1)

        # saving with shape and dtype
        shape = prototype.shape
        if prototype.dtype == np.float32:
            dtype = DType.float32
        else:
            raise RuntimeError("Test is not configured to run with another type")
        path = f'{time.time()}.onnx'
        save_sparkml(spark_model, path, shape=shape, dtype=dtype)
        model = load_model(path)
        os.remove(path)
        con.modelset('spark_model2', Backend.onnx, Device.cpu, model)
        con.tensorset('a2', in_tensor)
        con.modelrun('spark_model2', ['a2'], ['outfromspark2'])
        tensor = con.tensorget('outfromspark2')
        self.assertEqual(len(tensor.value), 1)

        # saving with initial_types
        inital_types = onnx_utils.get_tensortype(shape, dtype)
        path = f'{time.time()}.onnx'
        save_sparkml(spark_model, path, initial_types=[inital_types])
        model = load_model(path)
        os.remove(path)
        con.modelset('spark_model3', Backend.onnx, Device.cpu, model)
        con.tensorset('a3', in_tensor)
        con.modelrun('spark_model3', ['a3'], ['outfromspark3'])
        tensor = con.tensorget('outfromspark3')
        self.assertEqual(len(tensor.value), 1)
