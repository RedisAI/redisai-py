import pickle

from .client import Device, Backend
# TODO: Move this imports inside functions if it affects performance
try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass  # that's Okey if you don't have TF

try:
    import torch
except ModuleNotFoundError:
    pass  # it's Okey if you don't have PT either



class Model:
    
    def __init__(self, path, device=Device.cpu, inputs=[], outputs=[]):
        pass

    @classmethod
    def save(cls, obj, path, output_names=[], is_native=False):
        # TODO: perhpas make output_names optional
        # TODO: what if user doens't save as .pb/.pt and if we are relying on format
        # TODO: check directory file exist etc
        # TODO: accept filepath or file like object
        if issubclass(type(obj), tf.Session):
            # todo: do the checks
            cls._save_tf_graph(obj, path, output_names, is_native)
        elif issubclass(type(type(obj)), torch.jit.ScriptMeta):
            # TODO make a proper check above
            cls._save_pt_graph(obj, path, is_native)
        else:
            # TODO proper error message
            raise TypeError('Invalid Object. Need PyTorch/Tensorflow graphs')

    @classmethod
    def _save_tf_graph(cls, sess, path, output_names, is_native):
        graph_def = sess.graph_def
        # clearing device information
        for node in graph_def.node:
            node.device = ""
        frozen = tf.graph_util.convert_variables_to_constants(
            sess, graph_def, output_names)
        # TODO: configure the log dire - second param
        if is_native:
            tf.io.write_graph(frozen, '.', path, as_text=False)
            return
        outdict = cls._get_filled_dict(
            frozen.SerializeToString(),
            Backend.tf, output_names=output_names)
        cls._write_custom_model(outdict, path)

    @classmethod
    def _save_pt_graph(cls, graph, path, is_native):
        # todo check if we need to clear device for pytorch
        # todo do `eval()`
        if is_native:
            torch.jit.save(graph, path)
            return
        outdict = cls._get_filled_dict(
            graph.save_to_buffer(), Backend.torch)
        cls._write_custom_model(outdict, path)

    @staticmethod
    def _get_filled_dict(graph, backend, input_names=[], output_names=[]):
        # todo : better way to convert arguments to dict
        return {
            'graph': graph,
            'backend': backend,
            'input_names': input_names,
            'output_names': output_names}

    @staticmethod
    def _write_custom_model(outdict, path):
        with open(path, 'wb') as file:
            pickle.dump(outdict, file)

    @classmethod
    def load(cls, path):
        pass

