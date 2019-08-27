from unittest import TestCase
import numpy as np
import os.path
from redisai import Client, DType, Backend, Device, Tensor, BlobTensor
from ml2rt import load_model
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
        con.tensorset('x', (2, 3), dtype=DType.float)
        values = con.tensorget('x', as_type=Tensor)
        self.assertEqual([2, 3], values.value)

        con.tensorset('x', Tensor.scalar(DType.int32, 2, 3))
        values = con.tensorget('x', as_type=Tensor).value
        self.assertEqual([2, 3], values)
        meta = con.tensorget('x', meta_only=True)
        self.assertTrue('<Tensor(shape=[2] type=DType.int32) at ' in repr(meta))

        self.assertRaises(Exception, con.tensorset, 1)
        self.assertRaises(Exception, con.tensorset, 'x')

    def test_numpy_tensor(self):
        con = self.get_client()
        input_array = np.array([2, 3])
        con.tensorset('x', input_array)
        values1 = con.tensorget('x')
        self.assertTrue(np.allclose([2, 3], values1))
        self.assertEqual(values1.dtype, np.int64)
        self.assertEqual(values1.shape, (2,))
        self.assertTrue((np.allclose(values1, input_array)))

        values2 = con.tensorget('x', as_type=BlobTensor)
        self.assertTrue(np.allclose(input_array, values2.to_numpy()))
        self.assertTrue(np.allclose(values1, values2.to_numpy()))
        ret = con.tensorset('x', values2)
        self.assertEqual(ret, b'OK')

    def test_run_tf_model(self):
        model_path = os.path.join(MODEL_DIR, 'graph.pb')
        bad_model_path = os.path.join(MODEL_DIR, 'pt-minimal.pt')

        model_pb = load_model(model_path)
        wrong_model_pb = load_model(bad_model_path)

        con = self.get_client()
        con.modelset('m', Backend.tf, Device.cpu, model_pb,
                     inputs=['a', 'b'], outputs='mul')

        # wrong model
        self.assertRaises(ResponseError,
                          con.modelset, 'm', Backend.tf, Device.cpu,
                          wrong_model_pb,
                          inputs=['a', 'b'], outputs='mul')
        # missing inputs/outputs
        self.assertRaises(ValueError,
                          con.modelset, 'm', Backend.tf, Device.cpu,
                          wrong_model_pb)

        # wrong backend
        self.assertRaises(ResponseError,
                          con.modelset, 'm', Backend.torch, Device.cpu,
                          model_pb,
                          inputs=['a', 'b'], outputs='mul')

        con.tensorset('a', Tensor.scalar(DType.float, 2, 3))
        con.tensorset('b', Tensor.scalar(DType.float, 2, 3))
        con.modelrun('m', ['a', 'b'], 'c')
        tensor = con.tensorget('c')
        self.assertTrue(np.allclose([4, 9], tensor))
        model_det = con.modelget('m')
        self.assertTrue(model_det['backend'] == Backend.tf)
        self.assertTrue(model_det['device'] == Device.cpu)
        con.modeldel('m')
        self.assertRaises(ResponseError, con.modelget, 'm')

    def test_scripts(self):
        con = self.get_client()
        self.assertRaises(ResponseError, con.scriptset,
                          'ket', Device.cpu, 'return 1')
        script = r"""
def bar(a, b):
    return a + b
"""
        con.scriptset('ket', Device.cpu, script)
        con.tensorset('a', Tensor.scalar(DType.float, 2, 3))
        con.tensorset('b', Tensor.scalar(DType.float, 2, 3))
        # try with bad arguments:
        self.assertRaises(ResponseError,
                          con.scriptrun, 'ket', 'bar', inputs='a', outputs='c')
        con.scriptrun('ket', 'bar', inputs=['a', 'b'], outputs='c')
        tensor = con.tensorget('c', as_type=Tensor)
        self.assertEqual([4, 6], tensor.value)
        script_det = con.scriptget('ket')
        self.assertTrue(script_det['device'] == Device.cpu)
        self.assertTrue(script_det['script'] == script)
        self.assertTrue("def bar(a, b):" in script_det['script'])
        con.scriptdel('ket')
        self.assertRaises(ResponseError, con.scriptget, 'ket')

    def test_run_onnxml_model(self):
        mlmodel_path = os.path.join(MODEL_DIR, 'boston.onnx')
        onnxml_model = load_model(mlmodel_path)
        con = self.get_client()
        con.modelset("onnx_model", Backend.onnx, Device.cpu, onnxml_model)
        tensor = BlobTensor.from_numpy(np.ones((1, 13), dtype=np.float32))
        con.tensorset("input", tensor)
        con.modelrun("onnx_model", ["input"], ["output"])
        outtensor = con.tensorget("output", as_type=Tensor)
        self.assertEqual(int(outtensor.value[0]), 24)

    def test_run_onnxdl_model(self):
        # A PyTorch model that finds the square
        dlmodel_path = os.path.join(MODEL_DIR, 'findsquare.onnx')
        onnxdl_model = load_model(dlmodel_path)
        con = self.get_client()
        con.modelset("onnx_model", Backend.onnx, Device.cpu, onnxdl_model)
        tensor = BlobTensor.from_numpy(np.array((2, 3), dtype=np.float32))
        con.tensorset("input", tensor)
        con.modelrun("onnx_model", ["input"], ["output"])
        outtensor = con.tensorget("output")
        self.assertTrue(np.allclose(outtensor, [4.0, 9.0]))


# TODO: image/blob tests; more numpy tests..
