import pickle
import os
import warnings

from .client import Device, Backend

try:
    import tensorflow as tf
except (ModuleNotFoundError, ImportError):
    pass  # that's Okey if you don't have TF

try:
    import torch
except (ModuleNotFoundError, ImportError):
    pass  # it's Okey if you don't have PT either



class Model:

    __slots__ = ['graph', 'backend', 'device', 'inputs', 'outputs']
    
    def __init__(self, path, device=Device.cpu, inputs=None, outputs=None):
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
    def save(cls, obj, path: str, input=None, output=None, as_native=True):
        """
        Infer the backend (TF/PyTorch) by inspecting the class hierarchy
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
        if issubclass(type(obj), tf.Session):
            cls._save_tf_graph(obj, path, output, as_native)
        elif issubclass(type(type(obj)), torch.jit.ScriptMeta):
            # TODO Is there a better way to check this
            cls._save_pt_graph(obj, path, as_native)
        else:
            raise TypeError(('Invalid Object. '
                'Need traced graph or scripted graph from PyTorch or '
                'Session object from Tensorflow'))

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
    def _save_pt_graph(cls, graph, path, as_native):
        # TODO how to handle the cpu/gpu
        if as_native:
            if graph.training == True:
                warnings.warn(
                    'Graph is in training mode. Converting to evaluation mode')
                graph.eval()
            torch.jit.save(graph, path)
            return
        else:
            raise NotImplementedError('Saving non-native graph is not supported yet')

    @staticmethod
    def _get_filled_dict(graph, backend, input=None, output=None):
        return {
            'graph': graph,
            'backend': backend,
            'input': input,
            'output': output}

    @staticmethod
    def _write_custom_model(outdict, path):
        with open(path, 'wb') as file:
            pickle.dump(outdict, file)

    @classmethod
    def load(cls, path:str):
        """
        Return the binary data if saved with `as_native` otherwise return the dict
        that contains binary graph/model on `graph` key. Check `_get_filled_dict`
        for more details.
        :param path: File path from where the native model or the rai models are saved
        """
        with open(path, 'rb') as f:
            return f.read()
