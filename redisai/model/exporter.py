import os
import warnings
from typing import Collection

from .._util import is_installed
from . import onnx_utils


def save_tensorflow(sess, path: str, output: Collection[str]):
    """
    Serialize a tensorflow session object to disk using TF utilities.
    :param sess: Tensorflow session object.
    :param path: Path to which the object will be serialized
    :param output: List of output nodes, required for TF sess to serialize
    """

    # TODO: TF 1.14+ has issue with __spec__
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


def save_torch(graph, path: str):
    """
    Serialize a torchscript object to disk using PyTorch utilities.
    :param graph: torchscript object
    :param path: Path to which the object will be serialized
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


def save_onnx(graph, path: str):
    """
    Serialize an ONNX object to disk.
    :param graph: ONNX graph object
    :param path: Path to which the object will be serialized
    """
    with open(path, 'wb') as f:
        f.write(graph.SerializeToString())


def save_sklearn(model, path: str, initial_types=None, prototype=None, shape=None, dtype=None):
    """
    Convert a scikit-learn model to onnx first and then save it to disk using `save_onnx`.
    We use onnxmltool to do the conversion from scikit-learn to ONNX and currently not all the
    scikit-learn models are supported by onnxmltools. A list of supported models can be found
    in the documentation.
    :param model: Scikit-learn model
    :param path: Path to which the object will be serialized
    :param initial_types: a python list. Each element is a tuple of a variable name and a type
        defined in onnxconverter_common.data_types. If initial type is empty, we'll guess the
        required information from prototype or infer it by using shape and dtype.
    :param prototype: A numpy array that gives shape and type information. This is ignored if
        initial_types is not None
    :param shape: Shape of the input to the model. Ignored if initial_types or prototype is not None
    :param dtype: redisai.DType object which represents the type of the input to the model.
        Ignored if initial_types or prototype is not None
    """
    if not is_installed(['onnxmltools', 'skl2onnx', 'pandas']):
        raise RuntimeError('Please install onnxmltools, skl2onnx & pandas to use this feature.')
    from onnxmltools import convert_sklearn
    if initial_types is None:
        initial_types = [onnx_utils.guess_onnx_tensortype(prototype, shape, dtype)]
    if not isinstance(initial_types, list):
        raise TypeError((
            "`initial_types` has to be a list. "
            "If you have only one initial_type, put that into a list"))
    serialized = convert_sklearn(model, initial_types=initial_types)
    save_onnx(serialized, path)


def save_sparkml(
        model, path, initial_types=None, prototype=None,
        shape=None, dtype=None, spark_session=None):
    """
    Convert a spark model to onnx first and then save it to disk using `save_onnx`.
    We use onnxmltool to do the conversion from spark to ONNX and currently not all the
    spark models are supported by onnxmltools. A list of supported models can be found
    in the documentation.
    :param model: PySpark model object
    :param path: Path to which the object will be serialized
    :param initial_types: a python list. Each element is a tuple of a variable name and a type
        defined in onnxconverter_common.data_types. If initial type is empty, we'll guess the
        required information from prototype or infer it by using shape and dtype.
    :param prototype: A numpy array that gives shape and type information. This is ignored if
        initial_types is not None
    :param shape: Shape of the input to the model. Ignored if initial_types or prototype is not None
    :param dtype: redisai.DType object which represents the type of the input to the model.
        Ignored if initial_types or prototype is not None
    """
    if not is_installed(['onnxmltools', 'pyspark']):
        raise RuntimeError('Please install onnxmltools & pyspark to use this feature.')
    from onnxmltools import convert_sparkml
    if initial_types is None:
        initial_types = [onnx_utils.guess_onnx_tensortype(prototype, shape, dtype)]
    if not isinstance(initial_types, list):
        raise TypeError((
            "`initial_types` has to be a list. "
            "If you have only one initial_type, put that into a list"))
    # TODO: test issue with passing different datatype for numerical values
    # known issue: https://github.com/onnx/onnxmltools/tree/master/onnxmltools/convert/sparkml
    serialized = convert_sparkml(model, initial_types=initial_types, spark_session=spark_session)
    save_onnx(serialized, path)
