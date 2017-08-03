import os
import tensorflow as tf
import grpc
import logging

from grpc import RpcError
from predict_client.predict_pb2 import PredictRequest
from predict_client.prediction_service_pb2 import PredictionServiceStub

logger = logging.getLogger(__name__)


class PredictClient:

    def __init__(self, localhost, envhost, model_name, model_version, num_scores=0):
        if envhost and envhost in os.environ:
            self.host = os.environ[envhost]
        else:
            logger.warning('Model host not in env variable')
            self.host = localhost

        self.model_name = model_name
        self.model_version = model_version
        self.num_scores = num_scores

    def predict(self, request_data, request_timeout=10):

        logger.info('Sending request to tfserving model')
        logger.info('Model name: ' + str(self.model_name))
        logger.info('Model version: ' + str(self.model_version))
        logger.info('Host: ' + str(self.host))

        tensor_shape = request_data.shape

        if self.model_name == 'incv4' or self.model_name == 'res152':
            features_tensor_proto = tf.contrib.util.make_tensor_proto(request_data, shape=tensor_shape)
        else:
            features_tensor_proto = tf.contrib.util.make_tensor_proto(request_data,
                                                                      dtype=tf.float32, shape=tensor_shape)

        # Create gRPC client and request
        channel = grpc.insecure_channel(self.host)
        stub = PredictionServiceStub(channel)
        request = PredictRequest()

        request.model_spec.name = self.model_name

        if self.model_version > 0:
            request.model_spec.version.value = self.model_version

        request.inputs['inputs'].CopyFrom(features_tensor_proto)

        try:
            result = stub.Predict(request, timeout=request_timeout)
            logger.debug('Got scores with len: ' + str(len(list(result.outputs['scores'].float_val))))
            return list(result.outputs['scores'].float_val)
        except RpcError as e:
            logger.error(e)
            logger.warning('Prediciton failed. Returning empty predictions of length: ' + str(self.num_scores))
            return [0] * self.num_scores

if __name__ == '__main__':
    import cv2
    img = cv2.imread('/Users/slp/Development/side_model/data/test_mini/Image1714772_1.jpg')
    # client = PredictClient('localhost:9000', 'INCV3_HOSTNAME', 'incv3', 1, prediction_shape=(2048,))
    client = PredictClient('localhost:9001', 'INCV3_HOSTNAME', 'incv3', 1)
    pred = client.predict(img)
    print(pred)
