import tensorflow as tf
import logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils


class MockClient:
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

        self.input_tensor_name = self.input_tensor_info[signature_constants.CLASSIFY_INPUTS].name

        # Mock client only supports one input, named 'inputs', for now
        if not self.input_tensor_name:
            raise ValueError('Unable to find input tensor of model.'
                             'Expected signature_constants.CLASSIFY_INPUTS to be only input tensor.')

        self.output_tensor_keys = [k for k in self.output_tensor_info]

        # Run all output tensors
        if len(self.output_tensor_keys) == 0:
            raise ValueError('Unable to find any output tensors of model.')

        self.output_tensor_names = [self.output_tensor_info[k].name for k in self.output_tensor_keys]

    def predict(self, request_data, **kwargs):

        self.logger.info('Sending request to inmemory model')
        self.logger.info('Model path: ' + str(self.model_path))

        self.logger.debug('Running tensors: ' + str(self.output_tensor_names))

        feed_dict = {self.input_tensor_name: request_data}

        results = self.sess.run(self.output_tensor_names, feed_dict=feed_dict)

        return {key: result for key, result in zip(self.output_tensor_keys, results)}
