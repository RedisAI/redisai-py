import os
import warnings

from ._util import is_installed
from ._util import guess_onnx_dtype


def save_tensorflow(sess, path, output):
    """
    TODO: Docstring
    """

    # TODO: tf 1.14+ has issue with __spec__
    if not is_installed('tensorflow'):
        raise RuntimeError('Please install Tensorflow to use this feature.')
    import tensorflow as tf
    graph_def = sess.graph_def
    # clearing device information
    for node in graph_def.node:
        node.device = ""
    frozen = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, output)
    directory = os.path.dirname(path)
    file = os.path.basename(path)
    tf.io.write_graph(frozen, directory, file, as_text=False)
    return


def save_torch(graph, path):
    """
    TODO: Docstring
    """
    if not is_installed('torch'):
        raise RuntimeError('Please install PyTorch to use this feature.')
    import torch
    # TODO how to handle the cpu/gpu
    if graph.training is True:
        warnings.warn(
            'Graph is in training mode. Converting to evaluation mode')
        graph.eval()
    torch.jit.save(graph, path)
    return


def save_onnx(graph, path):
    """
    TODO: Docstring
    """
    with open(path, 'wb') as f:
        f.write(graph.SerializeToString())


def save_sklearn(graph, path, prototype):
    """
    TODO: Docstring
    """
    if not is_installed(['onnxmltools', 'skl2onnx', 'pandas']):
        raise RuntimeError('Please install onnxmltools, skl2onnx & pandas to use this feature.')
    from onnxmltools import convert_sklearn

    datatype = guess_onnx_dtype(prototype)
    serialized = convert_sklearn(graph, initial_types=datatype)
    save_onnx(serialized, path)


def save_sparkml(graph, path, prototype=None, shape=None, dtype=None):
    """
    TODO: Docstring
    """
    if not is_installed(['onnxmltools', 'pyspark']):
        raise RuntimeError('Please install onnxmltools & pyspark to use this feature.')
    from onnxmltools import convert_sparkml

    # TODO: test issue with passing different datatype for numerical values
    # known issue: https://github.com/onnx/onnxmltools/tree/master/onnxmltools/convert/sparkml
    datatype = guess_onnx_dtype(prototype)
    serialized = convert_sparkml(graph, initial_types=datatype)
    save_onnx(serialized, path)


def save_coreml(graph, path):
    if not is_installed(['onnxmltools', 'coremltools']):
        raise RuntimeError('Please install onnxmltools & coremltools to use this feature.')
    from onnxmltools import convert_coreml

    serialized = convert_coreml(graph)
    save_onnx(serialized, path)


def save_xgboost(graph, path, prototype):
    if not is_installed(['onnxmltools', 'xgboost']):
        raise RuntimeError('Please install onnxmltools & xgboost to use this feature.')
    from onnxmltools import convert_xgboost

    datatype = guess_onnx_dtype(prototype)
    serialized = convert_xgboost(graph, initial_types=datatype)
    save_onnx(serialized, path)


def load_model(path: str):
    """
    Return the binary data if saved with `as_native` otherwise return the dict
    that contains binary graph/model on `graph` key (Not implemented yet).
    :param path: File path from where the native model or the rai models are saved
    """
    with open(path, 'rb') as f:
        return f.read()
