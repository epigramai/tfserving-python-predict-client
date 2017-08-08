import tensorflow as tf
import grpc
import logging

from grpc import RpcError
from predict_client.predict_pb2 import PredictRequest
from predict_client.prediction_service_pb2 import PredictionServiceStub
from predict_client.abstract_client import AbstractPredictClient

logger = logging.getLogger(__name__)


class MockPredictClient(AbstractPredictClient):

    def __init__(self, host, model_name, model_version, num_scores=0):
        super().__init__(host, model_name, model_version, num_scores)

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
            logger.debug('Predicted scores with len: ' + str(len(list(result.outputs['scores'].float_val))))
            return list(result.outputs['scores'].float_val)
        except RpcError as e:
            logger.warning(e)
            logger.warning('Prediction failed. Mock client will return empty prediction of length: '
                           + str(self.num_scores))
            return [0] * self.num_scores
