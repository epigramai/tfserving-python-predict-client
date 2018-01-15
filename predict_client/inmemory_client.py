import tensorflow as tf
import logging
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils


class InMemoryClient:
    def __init__(self, model_path):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path

        if not tf.saved_model.loader.maybe_saved_model_directory(self.model_path):
            raise ValueError('No model found in', self.model_path)

        self.sess = tf.Session(graph=tf.Graph())

        meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.model_path)
        signature_def = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        meta_graph_def_sig = signature_def_utils.get_signature_def_by_key(meta_graph_def, signature_def)

        self.input_tensor_info = meta_graph_def_sig.inputs

        self.output_tensor_info = meta_graph_def_sig.outputs
        self.output_tensor_keys = [k for k in self.output_tensor_info]
        self.output_tensor_names = [self.output_tensor_info[k].name for k in self.output_tensor_keys]

    def predict(self, request_data, **kwargs):

        self.logger.info('Sending request to inmemory model')
        self.logger.info('Model path: ' + str(self.model_path))

        feed_dict = dict()
        for d in request_data:
            input_tensor_name = self.input_tensor_info[d['in_tensor_name']].name
            feed_dict[input_tensor_name] = d['data']

        results = self.sess.run(self.output_tensor_names, feed_dict=feed_dict)

        return {key: result for key, result in zip(self.output_tensor_keys, results)}
