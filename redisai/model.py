import os
import warnings
import sys

try:
    import tensorflow as tf
except (ModuleNotFoundError, ImportError):
    pass

try:
    import torch
except (ModuleNotFoundError, ImportError):
    pass

try:
    import onnx
except (ModuleNotFoundError, ImportError):
    pass

try:
    import skl2onnx
    import sklearn
except (ModuleNotFoundError, ImportError):
    pass


class Model:

    __slots__ = ['graph', 'backend', 'device', 'inputs', 'outputs']

    def __init__(self, path, device=None, inputs=None, outputs=None):
        """
        Declare a model suitable for passing to modelset
        :param path: Filepath from where the stored model can be read
        :param device: Enum from `redisai.Device` represents which device
            should the model run on, inside RedisAI
        :param inputs: Optional parameter required only for tensorflow.
            In the TF world, this represents the list which is being
            passed to `sess.run` with tensors which is required for
            TF to execute the model
        :param outputs: Optional parameter required only for tensorflow.
            Similr to `inputs`, `outputs` is also passed to `sess.run` but
            to fetch the output from
        """
        raise NotImplementedError('Instance creation is not impelemented yet')

    @classmethod
    def save(cls, obj, path: str, input=None, output=None, as_native=True, prototype=None):
        """
        Infer the backend (TF/PyTorch/ONNX) by inspecting the class hierarchy
        and calls the appropriate serialization utility. It is essentially a
        wrapper over serialization mechanism of each backend
        :param path: Path to which the graph/model will be saved
        :param input: Optional parameter required only for tensorflow.
            In the TF world, this represents the list which is being
            passed to `sess.run` with tensors which is required for
            TF to execute the model
        :param output: Optional parameter required only for tensorflow.
            Similr to `input`, `output` is also passed to `sess.run` but
            to fetch the output from
        :param as_native: Saves the graph/model with backend's serialization
            mechanism if True. If False, custom saving utility will be called
            which saves other informations required for modelset. Defaults to True
        """
        if 'tensorflow' in sys.modules and issubclass(type(obj), tf.Session):
            cls._save_tf_graph(obj, path, output, as_native)
        elif 'torch' in sys.modules and issubclass(
                type(type(obj)), torch.jit.ScriptMeta):
            # TODO Is there a better way to check this
            cls._save_torch_graph(obj, path, as_native)
        elif 'onnx' in sys.modules and issubclass(
                type(obj), onnx.onnx_ONNX_RELEASE_ml_pb2.ModelProto):
            cls._save_onnx_graph(obj, path, as_native)
        elif 'skl2onnx' in sys.modules and issubclass(
                type(obj), sklearn.base.BaseEstimator):
            cls._save_sklearn_graph(obj, path, as_native, prototype)
        else:
            message = ("Could not find the required dependancy to export the graph object. "
                       "`save_model` relies on serialization mechanism provided by the"
                       " supported backends such as Tensorflow, PyTorch, ONNX or skl2onnx. "
                       "Please install package required for serializing your graph. "
                       "For more information, checkout the redisia-py documentation")
            raise RuntimeError(message)

    @classmethod
    def _save_tf_graph(cls, sess, path, output, as_native):
        graph_def = sess.graph_def
        # clearing device information
        for node in graph_def.node:
            node.device = ""
        frozen = tf.graph_util.convert_variables_to_constants(
            sess, graph_def, output)
        if as_native:
            directory = os.path.dirname(path)
            file = os.path.basename(path)
            tf.io.write_graph(frozen, directory, file, as_text=False)
            return
        else:
            raise NotImplementedError('Saving non-native graph is not supported yet')

    @classmethod
    def _save_torch_graph(cls, graph, path, as_native):
        # TODO how to handle the cpu/gpu
        if as_native:
            if graph.training is True:
                warnings.warn(
                    'Graph is in training mode. Converting to evaluation mode')
                graph.eval()
            torch.jit.save(graph, path)
            return
        else:
            raise NotImplementedError('Saving non-native graph is not supported yet')

    @classmethod
    def _save_onnx_graph(cls, graph, path, as_native):
        if as_native:
            with open(path, 'wb') as f:
                f.write(graph.SerializeToString())
        else:
            raise NotImplementedError('Saving non-native graph is not supported yet')

    @classmethod
    def _save_sklearn_graph(cls, graph, path, as_native, prototype):
        if not as_native:
            raise NotImplementedError('Saving non-native graph is not supported yet')
        if hasattr(prototype, 'shape') and hasattr(prototype, 'dtype'):
            datatype = skl2onnx.common.data_types.guess_data_type(prototype)
            serialized = skl2onnx.convert_sklearn(graph, initial_types=datatype)
            cls._save_onnx_graph(serialized, path, as_native)
        else:
            raise TypeError(
                "Serializing scikit learn model needs to know shape and dtype"
                " of input data which will be inferred from `prototype` "
                "parameter. It has to be a valid `numpy.ndarray` of shape of your input")

    @classmethod
    def load(cls, path: str):
        """
        Return the binary data if saved with `as_native` otherwise return the dict
        that contains binary graph/model on `graph` key (Not implemented yet).
        :param path: File path from where the native model or the rai models are saved
        """
        with open(path, 'rb') as f:
            return f.read()
