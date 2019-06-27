import os
import warnings
import importlib
from . import onnx_utils


def is_installed(packages):
    if not isinstance(packages, list):
        packages = [packages]
    for p in packages:
        if importlib.util.find_spec(p) is None:
            return False
    return True


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


def save_sklearn(graph, path, initial_types=None, prototype=None, shape=None, dtype=None):
    """
    TODO: Docstring
    """
    if not is_installed(['onnxmltools', 'skl2onnx', 'pandas']):
        raise RuntimeError('Please install onnxmltools, skl2onnx & pandas to use this feature.')
    from onnxmltools import convert_sklearn
    if not initial_types:
        initial_types = [onnx_utils.guess_onnx_tensortype(prototype, shape, dtype)]
    if not isinstance(initial_types, list):
        raise TypeError((
            "`initial_types` has to be a list. "
            "If you have only one initial_type, put that into a list"))
    serialized = convert_sklearn(graph, initial_types=initial_types)
    save_onnx(serialized, path)


def save_sparkml(
        graph, path, initial_types=None, prototype=None,
        shape=None, dtype=None, spark_session=None):
    """
    TODO: Docstring
    """
    if not is_installed(['onnxmltools', 'pyspark']):
        raise RuntimeError('Please install onnxmltools & pyspark to use this feature.')
    from onnxmltools import convert_sparkml

    # TODO: test issue with passing different datatype for numerical values
    # known issue: https://github.com/onnx/onnxmltools/tree/master/onnxmltools/convert/sparkml
    serialized = convert_sparkml(graph, initial_types=initial_types, spark_session=spark_session)
    save_onnx(serialized, path)
