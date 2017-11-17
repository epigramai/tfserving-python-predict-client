import logging
import time
import tensorflow as tf
import grpc
from grpc import RpcError
from predict_client.pbs.prediction_service_pb2 import PredictionServiceStub
from predict_client.pbs.predict_pb2 import PredictRequest

from predict_client.util import result_to_dict

tf_dtype_mapping = {
    'uint8': tf.uint8,
    'float32': tf.float32
}


class ProdClient:
    def __init__(self, host, model_name, model_version, in_tensor_dtype='float32'):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.host = host
        self.model_name = model_name
        self.model_version = model_version

        if in_tensor_dtype not in tf_dtype_mapping:
            self.in_tensor_dtype = tf.float32
            self.logger.info('Param in_tensor_dtype not in tf_dype_mapping. Trying to use tf.float32.')
        else:
            self.in_tensor_dtype = tf_dtype_mapping[in_tensor_dtype]

    def predict(self, request_data, request_timeout=10):

        self.logger.info('Sending request to tfserving model')
        self.logger.info('Host: ' + str(self.host))
        self.logger.info('Model name: ' + str(self.model_name))
        self.logger.info('Model version: ' + str(self.model_version))

        t = time.time()
        self.logger.debug('Request data shape: ' + str(request_data.shape))
        tensor_proto = tf.contrib.util.make_tensor_proto(request_data, dtype=self.in_tensor_dtype,
                                                         shape=request_data.shape)

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
            self.logger.info('Got result')

            result_dict = result_to_dict(result)

            keys = [k for k in result_dict]
            self.logger.info('Got result with keys: ' + str(keys))

            return result_dict

        except RpcError as e:
            self.logger.error(e)
            self.logger.error('Prediction failed!')

        return {}
