from unittest import TestCase
import numpy as np
import os.path
from redisai import Client, DType, Backend, Device
from ml2rt import load_model
from redis.exceptions import ResponseError


MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + '/testdata'


class ClientTestCase(TestCase):
    def setUp(self):
        super(ClientTestCase, self).setUp()
        self.get_client().flushall()

    def get_client(self):
        return Client()

    def test_set_non_numpy_tensor(self):
        con = self.get_client()
        con.tensorset('x', (2, 3, 4, 5), dtype=DType.float)
        result = con.tensorget('x', as_numpy=False)
        self.assertEqual([2, 3, 4, 5], result.value)
        self.assertEqual((4,), result.shape)

        con.tensorset('x', (2, 3, 4, 5), dtype=DType.int16, shape=(2, 2))
        result = con.tensorget('x', as_numpy=False)
        self.assertEqual([2, 3, 4, 5], result.value)
        self.assertEqual((2, 2), result.shape)

        with self.assertRaises(AttributeError):
            con.tensorset('x', (2, 3, 4), dtype=DType.int)

        with self.assertRaises(TypeError):
            con.tensorset('x')
            con.tensorset(1)

    def test_meta(self):
        con = self.get_client()
        con.tensorset('x', (2, 3, 4, 5), dtype=DType.float)
        result = con.tensorget('x', meta_only=True)
        self.assertEqual([], result.value)
        self.assertEqual((4,), result.shape)

    def test_numpy_tensor(self):
        con = self.get_client()

        input_array = np.array([2, 3])
        con.tensorset('x', input_array)
        values = con.tensorget('x')
        self.assertTrue(np.allclose([2, 3], values))
        self.assertEqual(values.dtype, np.int64)
        self.assertEqual(values.shape, (2,))
        self.assertTrue((np.allclose(values, input_array)))
        ret = con.tensorset('x', values)
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

        con.tensorset('a', (2, 3), dtype=DType.float)
        con.tensorset('b', (2, 3), dtype=DType.float)
        con.modelrun('m', ['a', 'b'], 'c')
        tensor = con.tensorget('c')
        self.assertTrue(np.allclose([4, 9], tensor))
        model_det = con.modelget('m')
        self.assertTrue(model_det.backend == Backend.tf)
        self.assertTrue(model_det.device == Device.cpu)
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
        con.tensorset('a', (2, 3), dtype=DType.float)
        con.tensorset('b', (2, 3), dtype=DType.float)
        # try with bad arguments:
        self.assertRaises(ResponseError,
                          con.scriptrun, 'ket', 'bar', inputs='a', outputs='c')
        con.scriptrun('ket', 'bar', inputs=['a', 'b'], outputs='c')
        tensor = con.tensorget('c', as_numpy=False)
        self.assertEqual([4, 6], tensor.value)
        script_det = con.scriptget('ket')
        self.assertTrue(script_det.device == Device.cpu)
        self.assertTrue(script_det.script == script)
        self.assertTrue("def bar(a, b):" in script_det.script)
        con.scriptdel('ket')
        self.assertRaises(ResponseError, con.scriptget, 'ket')

    def test_run_onnxml_model(self):
        mlmodel_path = os.path.join(MODEL_DIR, 'boston.onnx')
        onnxml_model = load_model(mlmodel_path)
        con = self.get_client()
        con.modelset("onnx_model", Backend.onnx, Device.cpu, onnxml_model)
        tensor = np.ones((1, 13)).astype(np.float32)
        con.tensorset("input", tensor)
        con.modelrun("onnx_model", ["input"], ["output"])
        # tests `convert_to_num`
        outtensor = con.tensorget("output", as_numpy=False)
        self.assertEqual(int(float(outtensor.value[0])), 24)

    def test_run_onnxdl_model(self):
        # A PyTorch model that finds the square
        dlmodel_path = os.path.join(MODEL_DIR, 'findsquare.onnx')
        onnxdl_model = load_model(dlmodel_path)
        con = self.get_client()
        con.modelset("onnx_model", Backend.onnx, Device.cpu, onnxdl_model)
        tensor = np.array((2,)).astype(np.float32)
        con.tensorset("input", tensor)
        con.modelrun("onnx_model", ["input"], ["output"])
        outtensor = con.tensorget("output")
        self.assertTrue(np.allclose(outtensor, [4.0]))

    def test_run_pytorch_model(self):
        model_path = os.path.join(MODEL_DIR, 'pt-minimal.pt')
        ptmodel = load_model(model_path)
        con = self.get_client()
        con.modelset("pt_model", Backend.torch, Device.cpu, ptmodel)
        con.tensorset('a', [2, 3, 2, 3], shape=(2, 2), dtype=DType.float)
        con.tensorset('b', [2, 3, 2, 3], shape=(2, 2), dtype=DType.float)
        con.modelrun("pt_model", ["a", "b"], "output")
        output = con.tensorget('output', as_numpy=False)
        self.assertTrue(np.allclose(output.value, [4, 6, 4, 6]))

    def test_run_tflite_model(self):
        model_path = os.path.join(MODEL_DIR, 'mnist_model_quant.tflite')
        tflmodel = load_model(model_path)
        con = self.get_client()
        con.modelset("tfl_model", Backend.tflite, Device.cpu, tflmodel)
        img = np.random.random((1, 1, 28, 28)).astype(np.float)
        con.tensorset('img', img)
        con.modelrun("tfl_model", "img", ["output1", "output2"])
        output = con.tensorget('output1')
        self.assertTrue(np.allclose(output, [8]))
