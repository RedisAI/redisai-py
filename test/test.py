import os.path
import sys
from io import StringIO
from unittest import TestCase
from skimage.io import imread
from skimage.transform import resize

import numpy as np
from ml2rt import load_model
from redis.exceptions import ResponseError

from redisai import Client

DEBUG = False
tf_graph = "graph.pb"
torch_graph = "pt-minimal.pt"
dog_img = "dog.jpg"


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/testdata"
script = r"""
def bar(a, b):
    return a + b
"""

data_processing_script = r"""
def pre_process_3ch(image):
    return image.float().div(255).unsqueeze(0)

def pre_process_4ch(image):
    return image.float().div(255)[:,:,:-1].contiguous().unsqueeze(0)

def post_process(output):
    # tf model has 1001 classes, hence negative 1
    return output.max(1)[1] - 1

def ensemble(output0, output1):
    return (output0 + output1) * 0.5
"""

class RedisAITestBase(TestCase):
    def setUp(self):
        super().setUp()
        self.get_client().flushall()

    def get_client(self, debug=DEBUG):
        return Client(debug)


class ClientTestCase(RedisAITestBase):
    def test_set_non_numpy_tensor(self):
        con = self.get_client()
        con.tensorset("x", (2, 3, 4, 5), dtype="float")
        result = con.tensorget("x", as_numpy=False)
        self.assertEqual([2, 3, 4, 5], result["values"])
        self.assertEqual([4], result["shape"])

        con.tensorset("x", (2, 3, 4, 5), dtype="float64")
        result = con.tensorget("x", as_numpy=False)
        self.assertEqual([2, 3, 4, 5], result["values"])
        self.assertEqual([4], result["shape"])
        self.assertEqual("DOUBLE", result["dtype"])

        con.tensorset("x", (2, 3, 4, 5), dtype="int16", shape=(2, 2))
        result = con.tensorget("x", as_numpy=False)
        self.assertEqual([2, 3, 4, 5], result["values"])
        self.assertEqual([2, 2], result["shape"])

        with self.assertRaises(TypeError):
            con.tensorset("x", (2, 3, 4, 5), dtype="wrongtype", shape=(2, 2))
        con.tensorset("x", (2, 3, 4, 5), dtype="int8", shape=(2, 2))
        result = con.tensorget("x", as_numpy=False)
        self.assertEqual("INT8", result["dtype"])
        self.assertEqual([2, 3, 4, 5], result["values"])
        self.assertEqual([2, 2], result["shape"])
        self.assertIn("values", result)

        with self.assertRaises(TypeError):
            con.tensorset("x")
            con.tensorset(1)

    def test_tensorget_meta(self):
        con = self.get_client()
        con.tensorset("x", (2, 3, 4, 5), dtype="float")
        result = con.tensorget("x", meta_only=True)
        self.assertNotIn("values", result)
        self.assertEqual([4], result["shape"])

    def test_numpy_tensor(self):
        con = self.get_client()

        input_array = np.array([2, 3], dtype=np.float32)
        con.tensorset("x", input_array)
        values = con.tensorget("x")
        self.assertEqual(values.dtype, np.float32)

        input_array = np.array([2, 3], dtype=np.float64)
        con.tensorset("x", input_array)
        values = con.tensorget("x")
        self.assertEqual(values.dtype, np.float64)

        input_array = np.array([2, 3])
        con.tensorset("x", input_array)
        values = con.tensorget("x")

        self.assertTrue(np.allclose([2, 3], values))
        self.assertEqual(values.dtype, np.int64)
        self.assertEqual(values.shape, (2,))
        self.assertTrue((np.allclose(values, input_array)))
        ret = con.tensorset("x", values)
        self.assertEqual(ret, "OK")

        # By default tensorget returns immutable, unless as_numpy_mutable is set as True
        ret = con.tensorget("x")
        self.assertRaises(ValueError, np.put, ret, 0, 1)
        ret = con.tensorget("x", as_numpy_mutable=True)
        np.put(ret, 0, 1)
        self.assertEqual(ret[0], 1)

        stringarr = np.array("dummy")
        with self.assertRaises(TypeError):
            con.tensorset("trying", stringarr)

    # AI.MODELSET is deprecated by AI.MODELSTORE.
    def test_deprecated_modelset(self):
        model_path = os.path.join(MODEL_DIR, "graph.pb")
        model_pb = load_model(model_path)
        con = self.get_client()
        with self.assertRaises(ValueError):
            con.modelset(
                "m",
                "tf",
                "wrongdevice",
                model_pb,
                inputs=["a", "b"],
                outputs=["mul"],
                tag="v1.0",
            )
        with self.assertRaises(ValueError):
            con.modelset(
                "m",
                "wrongbackend",
                "cpu",
                model_pb,
                inputs=["a", "b"],
                outputs=["mul"],
                tag="v1.0",
            )
        con.modelset(
            "m", "tf", "cpu", model_pb, inputs=["a", "b"], outputs=["mul"], tag="v1.0"
        )
        model = con.modelget("m", meta_only=True)
        self.assertEqual(
            model,
            {
                "backend": "TF",
                "batchsize": 0,
                "device": "cpu",
                "inputs": ["a", "b"],
                "minbatchsize": 0,
                "minbatchtimeout": 0,
                "outputs": ["mul"],
                "tag": "v1.0",
            },
        )

    def test_modelstore_errors(self):
        model_path = os.path.join(MODEL_DIR, "graph.pb")
        model_pb = load_model(model_path)
        con = self.get_client()

        with self.assertRaises(ValueError) as e:
            con.modelstore(
                None,
                "TF",
                "CPU",
                model_pb,
                inputs=["a", "b"],
                outputs=["mul"]
            )
        self.assertEqual(str(e.exception), "Model name was not given")

        with self.assertRaises(ValueError) as e:
            con.modelstore(
                "m",
                "tf",
                "wrongdevice",
                model_pb,
                inputs=["a", "b"],
                outputs=["mul"],
                tag="v1.0",
            )
        self.assertTrue(str(e.exception).startswith("Device not allowed"))
        with self.assertRaises(ValueError) as e:
            con.modelstore(
                "m",
                "wrongbackend",
                "cpu",
                model_pb,
                inputs=["a", "b"],
                outputs=["mul"],
                tag="v1.0",
            )
        self.assertTrue(str(e.exception).startswith("Backend not allowed"))
        with self.assertRaises(ValueError) as e:
            con.modelstore(
                "m",
                "tf",
                "cpu",
                model_pb,
                inputs=["a", "b"],
                outputs=["mul"],
                tag="v1.0",
                minbatch=2,
            )
        self.assertEqual(str(e.exception),
                         "Minbatch is not allowed without batch")
        with self.assertRaises(ValueError) as e:
            con.modelstore(
                "m",
                "tf",
                "cpu",
                model_pb,
                inputs=["a", "b"],
                outputs=["mul"],
                tag="v1.0",
                batch=4,
                minbatchtimeout=1000,
            )
        self.assertTrue(
            str(e.exception), "Minbatchtimeout is not allowed without minbatch"
        )
        with self.assertRaises(ValueError) as e:
            con.modelstore("m", "tf", "cpu", model_pb, tag="v1.0")
        self.assertTrue(
            str(e.exception),
            "Require keyword arguments inputs and outputs for TF models",
        )
        with self.assertRaises(ValueError) as e:
            con.modelstore(
                "m",
                "torch",
                "cpu",
                model_pb,
                inputs=["a", "b"],
                outputs=["mul"],
                tag="v1.0",
            )
        self.assertTrue(
            str(e.exception),
            "Inputs and outputs keywords should not be specified for this backend",
        )

    def test_modelget_meta(self):
        model_path = os.path.join(MODEL_DIR, tf_graph)
        model_pb = load_model(model_path)
        con = self.get_client()
        con.modelstore(
            "m", "tf", "cpu", model_pb, inputs=["a", "b"], outputs=["mul"], tag="v1.0"
        )
        model = con.modelget("m", meta_only=True)
        self.assertEqual(
            model,
            {
                "backend": "TF",
                "batchsize": 0,
                "device": "cpu",
                "inputs": ["a", "b"],
                "minbatchsize": 0,
                "minbatchtimeout": 0,
                "outputs": ["mul"],
                "tag": "v1.0",
            },
        )

    def test_modelexecute_non_list_input_output(self):
        model_path = os.path.join(MODEL_DIR, "graph.pb")
        model_pb = load_model(model_path)
        con = self.get_client()
        con.modelstore(
            "m", "tf", "cpu", model_pb, inputs=["a", "b"], outputs=["mul"], tag="v1.7"
        )
        con.tensorset("a", (2, 3), dtype="float")
        con.tensorset("b", (2, 3), dtype="float")
        ret = con.modelexecute("m", ["a", "b"], "out")
        self.assertEqual(ret, "OK")

    def test_nonasciichar(self):
        nonascii = "ĉ"
        model_path = os.path.join(MODEL_DIR, tf_graph)
        model_pb = load_model(model_path)
        con = self.get_client()
        con.modelstore(
            "m" + nonascii,
            "tf",
            "cpu",
            model_pb,
            inputs=["a", "b"],
            outputs=["mul"],
            tag="v1.0",
        )
        con.tensorset("a" + nonascii, (2, 3), dtype="float")
        con.tensorset("b", (2, 3), dtype="float")
        con.modelexecute(
            "m" + nonascii, ["a" + nonascii, "b"], ["c" + nonascii])
        tensor = con.tensorget("c" + nonascii)
        self.assertTrue((np.allclose(tensor, [4.0, 9.0])))

    def test_run_tf_model(self):
        model_path = os.path.join(MODEL_DIR, tf_graph)
        bad_model_path = os.path.join(MODEL_DIR, torch_graph)

        model_pb = load_model(model_path)
        wrong_model_pb = load_model(bad_model_path)

        con = self.get_client()
        con.modelstore(
            "m", "tf", "cpu", model_pb, inputs=["a", "b"], outputs=["mul"], tag="v1.0"
        )
        con.modeldel("m")
        self.assertRaises(ResponseError, con.modelget, "m")
        con.modelstore(
            "m", "tf", "cpu", model_pb, inputs=["a", "b"], outputs="mul", tag="v1.0"
        )

        # Required arguments ar None
        with self.assertRaises(ValueError) as e:
            con.modelexecute(
                "m",
                inputs=None,
                outputs=None
            )
        self.assertEqual(str(e.exception), "Missing required arguments for model execute command")

        # wrong model
        with self.assertRaises(ResponseError) as e:
            con.modelstore(
                "m", "tf", "cpu", wrong_model_pb, inputs=["a", "b"], outputs=["mul"]
            )
        self.assertEqual(str(e.exception), "Invalid GraphDef")

        con.tensorset("a", (2, 3), dtype="float")
        con.tensorset("b", (2, 3), dtype="float")
        con.modelexecute("m", ["a", "b"], ["c"])
        tensor = con.tensorget("c")
        self.assertTrue(np.allclose([4, 9], tensor))
        model_det = con.modelget("m")
        self.assertTrue(model_det["backend"] == "TF")
        self.assertTrue(
            model_det["device"] == "cpu"
        )  # TODO; RedisAI returns small letter
        self.assertTrue(model_det["tag"] == "v1.0")
        con.modeldel("m")
        self.assertRaises(ResponseError, con.modelget, "m")

    def test_scripts(self):
        con = self.get_client()
        self.assertRaises(ResponseError, con.scriptset,
                          "ket", "cpu", "return 1")
        con.scriptset("ket", "cpu", script)
        con.tensorset("a", (2, 3), dtype="float")
        con.tensorset("b", (2, 3), dtype="float")
        # try with bad arguments:
        self.assertRaises(
            ResponseError, con.scriptrun, "ket", "bar", inputs=["a"], outputs=["c"]
        )
        con.scriptrun("ket", "bar", inputs=["a", "b"], outputs=["c"])
        tensor = con.tensorget("c", as_numpy=False)
        self.assertEqual([4, 6], tensor["values"])
        script_det = con.scriptget("ket")
        self.assertTrue(script_det["device"] == "cpu")
        self.assertTrue(script_det["source"] == script)
        script_det = con.scriptget("ket", meta_only=True)
        self.assertTrue(script_det["device"] == "cpu")
        self.assertNotIn("source", script_det)
        con.scriptdel("ket")
        self.assertRaises(ResponseError, con.scriptget, "ket")

    def test_run_onnxml_model(self):
        mlmodel_path = os.path.join(MODEL_DIR, "boston.onnx")
        onnxml_model = load_model(mlmodel_path)
        con = self.get_client()
        con.modelstore("onnx_model", "onnx", "cpu", onnxml_model)
        tensor = np.ones((1, 13)).astype(np.float32)
        con.tensorset("input", tensor)
        con.modelexecute("onnx_model", ["input"], ["output"])
        # tests `convert_to_num`
        outtensor = con.tensorget("output", as_numpy=False)
        self.assertEqual(int(float(outtensor["values"][0])), 24)

    def test_run_onnxdl_model(self):
        # A PyTorch model that finds the square
        dlmodel_path = os.path.join(MODEL_DIR, "findsquare.onnx")
        onnxdl_model = load_model(dlmodel_path)
        con = self.get_client()
        con.modelstore("onnx_model", "onnx", "cpu", onnxdl_model)
        tensor = np.array((2,)).astype(np.float32)
        con.tensorset("input", tensor)
        con.modelexecute("onnx_model", ["input"], ["output"])
        outtensor = con.tensorget("output")
        self.assertTrue(np.allclose(outtensor, [4.0]))

    def test_run_pytorch_model(self):
        model_path = os.path.join(MODEL_DIR, torch_graph)
        ptmodel = load_model(model_path)
        con = self.get_client()
        con.modelstore("pt_model", "torch", "cpu", ptmodel, tag="v1.0")
        con.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        con.tensorset("b", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        con.modelexecute("pt_model", ["a", "b"], ["output"])
        output = con.tensorget("output", as_numpy=False)
        self.assertTrue(np.allclose(output["values"], [4, 6, 4, 6]))

    def test_run_tflite_model(self):
        model_path = os.path.join(MODEL_DIR, "mnist_model_quant.tflite")
        tflmodel = load_model(model_path)
        con = self.get_client()
        con.modelstore("tfl_model", "tflite", "cpu", tflmodel)
        img = np.random.random((1, 1, 28, 28)).astype(np.float)
        con.tensorset("img", img)
        con.modelexecute("tfl_model", ["img"], ["output1", "output2"])
        output = con.tensorget("output1")
        self.assertTrue(np.allclose(output, [8]))

    # AI.MODELRUN is deprecated by AI.MODELEXECUTE
    def test_deprecated_modelrun(self):
        model_path = os.path.join(MODEL_DIR, "graph.pb")
        model_pb = load_model(model_path)

        con = self.get_client()
        con.modelstore(
            "m", "tf", "cpu", model_pb, inputs=["a", "b"], outputs=["mul"], tag="v1.0"
        )

        con.tensorset("a", (2, 3), dtype="float")
        con.tensorset("b", (2, 3), dtype="float")
        con.modelrun("m", ["a", "b"], ["c"])
        tensor = con.tensorget("c")
        self.assertTrue(np.allclose([4, 9], tensor))

    def test_info(self):
        model_path = os.path.join(MODEL_DIR, tf_graph)
        model_pb = load_model(model_path)
        con = self.get_client()
        con.modelstore("m", "tf", "cpu", model_pb,
                       inputs=["a", "b"], outputs=["mul"])
        first_info = con.infoget("m")
        expected = {
            "key": "m",
            "type": "MODEL",
            "backend": "TF",
            "device": "cpu",
            "tag": "",
            "duration": 0,
            "samples": 0,
            "calls": 0,
            "errors": 0,
        }
        self.assertEqual(first_info, expected)
        con.tensorset("a", (2, 3), dtype="float")
        con.tensorset("b", (2, 3), dtype="float")
        con.modelexecute("m", ["a", "b"], ["c"])
        con.modelexecute("m", ["a", "b"], ["c"])
        second_info = con.infoget("m")
        self.assertEqual(second_info["calls"], 2)  # 2 model runs
        con.inforeset("m")
        third_info = con.infoget("m")
        # before modelrun and after reset
        self.assertEqual(first_info, third_info)

    def test_model_scan(self):
        model_path = os.path.join(MODEL_DIR, tf_graph)
        model_pb = load_model(model_path)
        con = self.get_client()
        con.modelstore(
            "m", "tf", "cpu", model_pb, inputs=["a", "b"], outputs=["mul"], tag="v1.2"
        )
        model_path = os.path.join(MODEL_DIR, "pt-minimal.pt")
        ptmodel = load_model(model_path)
        con = self.get_client()
        # TODO: RedisAI modelscan issue
        con.modelstore("pt_model", "torch", "cpu", ptmodel)
        mlist = con.modelscan()
        self.assertEqual(mlist, [["pt_model", ""], ["m", "v1.2"]])

    def test_script_scan(self):
        con = self.get_client()
        con.scriptset("ket1", "cpu", script, tag="v1.0")
        con.scriptset("ket2", "cpu", script)
        slist = con.scriptscan()
        self.assertEqual(slist, [["ket1", "v1.0"], ["ket2", ""]])

    def test_debug(self):
        con = self.get_client(debug=True)
        with Capturing() as output:
            con.tensorset("x", (2, 3, 4, 5), dtype="float")
        self.assertEqual(["AI.TENSORSET x FLOAT 4 VALUES 2 3 4 5"], output)


def load_image():
    image_filename = os.path.join(MODEL_DIR, dog_img)
    img_height, img_width = 224, 224

    img = imread(image_filename)
    img = resize(img, (img_height, img_width), mode='constant', anti_aliasing=True)
    img = img.astype(np.uint8)
    return img


class DagTestCase(RedisAITestBase):
    def setUp(self):
        super().setUp()
        con = self.get_client()
        model_path = os.path.join(MODEL_DIR, torch_graph)
        ptmodel = load_model(model_path)
        con.modelstore("pt_model", "torch", "cpu", ptmodel, tag="v7.0")

    def test_deprecated_dugrun(self):
        con = self.get_client()
        con.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        con.tensorset("b", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        dag = con.dag(load=["a", "b"], persist="output")
        dag.modelrun("pt_model", ["a", "b"], ["output"])
        dag.tensorget("output")
        result = dag.run()
        expected = ["OK", np.array([[4.0, 6.0], [4.0, 6.0]], dtype=np.float32)]
        result_outside_dag = con.tensorget("output")
        self.assertTrue(np.allclose(expected.pop(), result.pop()))
        result = dag.run()
        self.assertTrue(np.allclose(result_outside_dag, result.pop()))
        self.assertEqual(expected, result)

    """
    def test_dagexecute_modelexecute_with_scriptexecute(self):
        con = self.get_client()
        script_name = 'imagenet_script:{1}'
        model_name = 'imagenet_model:{1}'
        
        img = load_image()
        model_path = os.path.join(MODEL_DIR, "resnet50.pb")
        model = load_model(model_path)
        con.scriptset(script_name, 'cpu', data_processing_script)
        con.modelstore(model_name, 'TF', 'cpu', model, inputs='images', outputs='output')
        
        dag = con.dag(persist='output:{1}')
        dag.tensorset('image:{1}', tensor=img, shape=(img.shape[1], img.shape[0]), dtype='UINT8')
        dag.scriptexecute(script_name, 'pre_process_3ch', keys=[], inputs='image:{1}', outputs='temp_key1')
        dag.modelexecute(model_name, inputs='temp_key1', outputs='temp_key2')
        dag.scriptexecute(script_name, 'post_process', keys=[], inputs='temp_key2', outputs='output:{1}')
        ret = dag.execute()
        self.assertEqual(['OK', 'OK', 'OK', 'OK'], ret)
    """

    def test_dagexecute_with_load(self):
        con = self.get_client()
        con.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float")

        dag = con.dag(load="a", keys="b")
        dag.tensorset("b", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        dag.modelexecute("pt_model", ["a", "b"], ["output"])
        dag.tensorget("output")
        result = dag.execute()
        expected = ["OK", "OK", np.array(
            [[4.0, 6.0], [4.0, 6.0]], dtype=np.float32)]
        self.assertTrue(np.allclose(expected.pop(), result.pop()))
        self.assertEqual(expected, result)
        self.assertRaises(ResponseError, con.tensorget, "b")

    def test_dagexecute_with_persist(self):
        con = self.get_client()

        with self.assertRaises(ResponseError):
            dag = con.dag(persist="wrongkey")
            dag.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float").execute()

        dag = con.dag(persist=["b"])
        dag.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        dag.tensorset("b", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        dag.tensorget("b")
        result = dag.execute()
        b = con.tensorget("b")
        self.assertTrue(np.allclose(b, result[-1]))
        self.assertEqual(b.dtype, np.float32)
        self.assertEqual(len(result), 3)

    def test_dagexecute_calling_on_return(self):
        con = self.get_client()
        con.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        result = (
            con.dag(load="a")
            .tensorset("b", [2, 3, 2, 3], shape=(2, 2), dtype="float")
            .modelexecute("pt_model", ["a", "b"], ["output"])
            .tensorget("output")
            .execute()
        )
        expected = ["OK", "OK", np.array(
            [[4.0, 6.0], [4.0, 6.0]], dtype=np.float32)]
        self.assertTrue(np.allclose(expected.pop(), result.pop()))
        self.assertEqual(expected, result)

    def test_dagexecute_without_load_and_persist(self):
        con = self.get_client()
        with self.assertRaises(RuntimeError):
            con.dag()

        dag = con.dag(load="wrongkey")
        with self.assertRaises(ResponseError):
            dag.tensorget("wrongkey").execute()

        dag = con.dag(keys=["a", "b", "output"])
        dag.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        dag.tensorset("b", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        dag.modelexecute("pt_model", ["a", "b"], ["output"])
        dag.tensorget("output")
        result = dag.execute()
        expected = [
            "OK",
            "OK",
            "OK",
            np.array([[4.0, 6.0], [4.0, 6.0]], dtype=np.float32),
        ]
        self.assertTrue(np.allclose(expected.pop(), result.pop()))
        self.assertEqual(expected, result)

    def test_dagexecute_with_load_and_persist(self):
        con = self.get_client()
        con.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        con.tensorset("b", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        dag = con.dag(load=["a", "b"], persist="output")
        dag.modelexecute("pt_model", ["a", "b"], ["output"])
        dag.tensorget("output")
        result = dag.execute()
        expected = ["OK", np.array([[4.0, 6.0], [4.0, 6.0]], dtype=np.float32)]
        result_outside_dag = con.tensorget("output")
        self.assertTrue(np.allclose(expected.pop(), result.pop()))
        result = dag.execute()
        self.assertTrue(np.allclose(result_outside_dag, result.pop()))
        self.assertEqual(expected, result)

    def test_dagexecuteRO(self):
        con = self.get_client()
        con.tensorset("a", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        con.tensorset("b", [2, 3, 2, 3], shape=(2, 2), dtype="float")
        with self.assertRaises(RuntimeError):
            con.dag(load=["a", "b"], persist="output", readonly=True)
        dag = con.dag(load=["a", "b"], readonly=True)
        """
        with self.assertRaises(RuntimeError):
            dag.scriptexecute()
        """
        dag.modelexecute("pt_model", ["a", "b"], ["output"])
        dag.tensorget("output")
        result = dag.execute()
        expected = ["OK", np.array([[4.0, 6.0], [4.0, 6.0]], dtype=np.float32)]
        self.assertTrue(np.allclose(expected.pop(), result.pop()))


class PipelineTest(RedisAITestBase):
    def test_pipeline_non_transaction(self):
        con = self.get_client()
        arr = np.array([[2.0, 3.0], [2.0, 3.0]], dtype=np.float32)
        pipe = con.pipeline(transaction=False)
        pipe = pipe.tensorset("a", arr).set("native", 1)
        pipe = pipe.tensorget("a", as_numpy=False)
        pipe = pipe.tensorget("a", as_numpy=True).tensorget(
            "a", meta_only=True)
        result = pipe.execute()
        expected = [
            b"OK",
            True,
            {"dtype": "FLOAT", "shape": [2, 2],
                "values": [2.0, 3.0, 2.0, 3.0]},
            arr,
            {"dtype": "FLOAT", "shape": [2, 2]},
        ]
        for res, exp in zip(result, expected):
            if isinstance(res, np.ndarray):
                self.assertTrue(np.allclose(exp, res))
            else:
                self.assertEqual(res, exp)

    def test_pipeline_transaction(self):
        con = self.get_client()
        arr = np.array([[2.0, 3.0], [2.0, 3.0]], dtype=np.float32)
        pipe = con.pipeline(transaction=True)
        pipe = pipe.tensorset("a", arr).set("native", 1)
        pipe = pipe.tensorget("a", as_numpy=False)
        pipe = pipe.tensorget("a", as_numpy=True).tensorget(
            "a", meta_only=True)
        result = pipe.execute()
        expected = [
            b"OK",
            True,
            {"dtype": "FLOAT", "shape": [2, 2],
                "values": [2.0, 3.0, 2.0, 3.0]},
            arr,
            {"dtype": "FLOAT", "shape": [2, 2]},
        ]
        for res, exp in zip(result, expected):
            if isinstance(res, np.ndarray):
                self.assertTrue(np.allclose(exp, res))
            else:
                self.assertEqual(res, exp)
