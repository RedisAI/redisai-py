from unittest import TestCase
import numpy as np
import os.path
from redisai import Client, DType, Backend, Device, Tensor, BlobTensor
from redis.exceptions import ResponseError


MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + '/testdata'


class TensorTestCase(TestCase):
    def testTensorShapes(self):
        t = Tensor(DType.float, [4], [1, 2, 3, 4])
        self.assertEqual([4], t.shape)
        t = BlobTensor.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual([2, 3], t.shape)


class ClientTestCase(TestCase):
    def setUp(self):
        super(ClientTestCase, self).setUp()
        self.get_client().flushall()

    def get_client(self):
        return Client()

    def test_set_tensor(self):
        con = self.get_client()
        con.tensorset('x', Tensor.scalar(DType.float, 2, 3))
        values = con.tensorget('x')
        self.assertEqual([2, 3], values.value)

        con.tensorset('x', Tensor.scalar(DType.int32, 2, 3))
        values = con.tensorget('x').value
        self.assertEqual([2, 3], values)

        self.assertRaises(Exception, con.tensorset, 1)
        self.assertRaises(Exception, con.tensorset, 'x')

    def test_numpy_tensor(self):
        con = self.get_client()
        con.tensorset('x', np.array([2, 3]))
        values = con.tensorget('x').value
        self.assertEqual([2, 3], values)

    def test_run_tf_model(self):
        model = os.path.join(MODEL_DIR, 'graph.pb')
        bad_model = os.path.join(MODEL_DIR, 'pt-minimal.pt')

        with open(model, 'rb') as f:
            model_pb = f.read()

        with open(bad_model, 'rb') as f:
            wrong_model_pb = f.read()

        con = self.get_client()
        con.modelset('m', Backend.tf, Device.cpu, model_pb,
                     input=['a', 'b'], output='mul')

        # wrong model
        self.assertRaises(ResponseError,
                          con.modelset, 'm', Backend.tf, Device.cpu,
                          wrong_model_pb,
                          input=['a', 'b'], output='mul')
        # missing inputs/outputs
        self.assertRaises(ValueError,
                          con.modelset, 'm', Backend.tf, Device.cpu,
                          wrong_model_pb)

        # wrong backend
        self.assertRaises(ResponseError,
                          con.modelset, 'm', Backend.torch, Device.cpu,
                          model_pb,
                          input=['a', 'b'], output='mul')

        con.tensorset('a', Tensor.scalar(DType.float, 2, 3))
        con.tensorset('b', Tensor.scalar(DType.float, 2, 3))
        con.modelrun('m', ['a', 'b'], 'c')
        tensor = con.tensorget('c')
        self.assertEqual([4, 9], tensor.value)

    def test_scripts(self):
        con = self.get_client()
        self.assertRaises(ResponseError, con.scriptset,
                          'ket', Device.cpu, 'return 1')
        con.scriptset('ket', Device.cpu, r"""
def bar(a, b):
    return a + b
""")
        con.tensorset('a', Tensor.scalar(DType.float, 2, 3))
        con.tensorset('b', Tensor.scalar(DType.float, 2, 3))
        # try with bad arguments:
        self.assertRaises(ResponseError,
                          con.scriptrun, 'ket', 'bar', input='a', output='c')
        con.scriptrun('ket', 'bar', input=['a', 'b'], output='c')
        tensor = con.tensorget('c')
        self.assertEqual([4, 6], tensor.value)


# TODO: image/blob tests; more numpy tests..