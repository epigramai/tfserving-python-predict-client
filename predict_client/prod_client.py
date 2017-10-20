import tensorflow as tf
import grpc
import logging
import numpy as np

from grpc import RpcError
from predict_client.predict_pb2 import PredictRequest
from predict_client.prediction_service_pb2 import PredictionServiceStub
import time

logger = logging.getLogger(__name__)

tf_dtype_mapping = {
    'uint8': tf.uint8,
    'float32': tf.float32
}


class PredictClient:
    def __init__(self, host, model_name, model_version, in_tensor_dtype='float32'):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.host = host
        self.model_name = model_name
        self.model_version = model_version

        if in_tensor_dtype not in tf_dtype_mapping:
            self.in_tensor_dtype = tf.float32
            logger.info('Param in_tensor_dtype not in tf_dype_mapping. Trying to use tf.float32.')
        else:
            self.in_tensor_dtype = tf_dtype_mapping[in_tensor_dtype]

    def predict(self, request_data, request_timeout=10):

        self.logger.info('Sending request to tfserving model')
        self.logger.info('Host: ' + str(self.host))
        self.logger.info('Model name: ' + str(self.model_name))
        self.logger.info('Model version: ' + str(self.model_version))

        t = time.time()
        logger.debug('Request data shape: ' + str(request_data.shape))
        tensor_proto = tf.contrib.util.make_tensor_proto(request_data, dtype=self.in_tensor_dtype, shape=request_data.shape)

        self.logger.debug('Making tensor proto took: ' + str(time.time() - t))

        # Create gRPC client and request
        t = time.time()
        channel = grpc.insecure_channel(self.host)
        self.logger.debug('Establishing insecure channel took: ' + str(time.time() - t))

        t = time.time()
        stub = PredictionServiceStub(channel)
        self.logger.debug('Creating stub took: ' + str(time.time() - t))

        t = time.time()
        request = PredictRequest()
        self.logger.debug('Creating request object took: ' + str(time.time() - t))

        request.model_spec.name = self.model_name

        if self.model_version > 0:
            request.model_spec.version.value = self.model_version

        request.inputs['inputs'].CopyFrom(tensor_proto)

        try:
            t = time.time()
            result = stub.Predict(request, timeout=request_timeout)

            self.logger.debug('Actual request took: ' + str(time.time() - t))

            # Model server returns a flat list of scores,
            # put all scores in an array with the correct shape
            score_shape = [x.size for x in result.outputs['scores'].tensor_shape.dim]
            output_scores = np.array(result.outputs['scores'].float_val).reshape(score_shape)

            classes_shape = [x.size for x in result.outputs['classes'].tensor_shape.dim]
            output_classes = np.array([c.decode('utf-8') for c in result.outputs['classes'].string_val])

            self.logger.info('Got scores with shape: ' + str(score_shape))
            self.logger.info('Got classes with shape: ' + str(classes_shape))

            return output_scores, output_classes
        except RpcError as e:
            self.logger.error(e)
            self.logger.error('Prediction failed!')
