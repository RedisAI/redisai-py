import os
import warnings

from ._util import is_installed
from ._util import guess_onnx_dtype


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
        raise DeprecationWarning((
            "Single entry for saving different object is deprecated. "
            "Use specific utility for each type. For more information checkout the documentation"))
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


def save_tensorflow(sess, path, output, as_native=True):
    """
    TODO: Docstring
    """
    if not is_installed('tensorflow'):
        raise RuntimeError('Please install Tensorflow to use this feature.')
    import tensorflow as tf
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


def save_torch(graph, path, as_native=True):
    """
    TODO: Docstring
    """
    if not is_installed('torch'):
        raise RuntimeError('Please install PyTorch to use this feature.')
    import torch
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


def save_onnx(graph, path, as_native=True):
    """
    TODO: Docstring
    """
    if as_native:
        with open(path, 'wb') as f:
            f.write(graph.SerializeToString())
    else:
        raise NotImplementedError('Saving non-native graph is not supported yet')


def save_sklearn(graph, path, prototype, as_native=True):
    """
    TODO: Docstring
    """
    if not is_installed(['onnxmltools', 'skl2onnx', 'pandas']):
        raise RuntimeError('Please install onnxmltools, skl2onnx & pandas to use this feature.')
    from onnxmltools import convert_sklearn

    if not as_native:
        raise NotImplementedError('Saving non-native graph is not supported yet')

    datatype = guess_onnx_dtype(prototype)
    serialized = convert_sklearn(graph, initial_types=datatype)
    save_onnx(serialized, path, as_native)


def save_sparkml(graph, path, prototype, as_native=True):
    """
    TODO: Docstring
    """
    if not is_installed(['onnxmltools', 'pyspark']):
        raise RuntimeError('Please install onnxmltools & pyspark to use this feature.')
    from onnxmltools import convert_sparkml

    if as_native:
        raise NotImplementedError('Saving non-native graph is not supported yet')

    # TODO: test issue with passing different datatype for numerical values
        # known issue: https://github.com/onnx/onnxmltools/tree/master/onnxmltools/convert/sparkml
    datatype = guess_onnx_dtype(prototype)
    serialized = convert_sparkml(graph, initial_types=datatype)
    save_onnx(serialized, path, as_native)


def save_coreml(graph, path, as_native=True):
    if not is_installed(['onnxmltools', 'coremltools']):
        raise RuntimeError('Please install onnxmltools & coremltools to use this feature.')
    from onnxmltools import convert_coreml

    if as_native:
        raise NotImplementedError('Saving non-native graph is not supported yet')
    serialized = convert_coreml(graph)
    save_onnx(serialized, path, as_native)


def save_xgboost(graph, path, prototype, as_native=True):
    if not is_installed(['onnxmltools', 'xgboost']):
        raise RuntimeError('Please install onnxmltools & xgboost to use this feature.')
    from onnxmltools import convert_xgboost

    if as_native:
        raise NotImplementedError('Saving non-native graph is not supported yet')
    datatype = guess_onnx_dtype(prototype)
    serialized = convert_xgboost(graph, initial_types=datatype)
    save_onnx(serialized, path, as_native)


def load_model(path: str):
    """
    Return the binary data if saved with `as_native` otherwise return the dict
    that contains binary graph/model on `graph` key (Not implemented yet).
    :param path: File path from where the native model or the rai models are saved
    """
    with open(path, 'rb') as f:
        return f.read()
