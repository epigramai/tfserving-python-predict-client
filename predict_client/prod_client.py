import logging
import time
from typing import List, Dict, Any, Tuple

import grpc
from grpc import RpcError
from predict_client.pbs.prediction_service_pb2 import PredictionServiceStub
from predict_client.pbs.predict_pb2 import PredictRequest
from predict_client.util import predict_response_to_dict, make_tensor_proto


class ProdClient:
    def __init__(self, host: str, model_name: str, model_version: int, options: List[Tuple[str, Any]] = None):
        """
        Args:
            host: The address and port of the model server

            model_name: The model name at the server

            model_version: The model version at the server

            options: grpc options passed to the channel

        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.host = host
        self.model_name = model_name
        self.model_version = model_version
        self.options = options

    def predict(self, request_data: List[Dict[str, Any]], request_timeout: int = 10):
        """ Get a model prediction on request data

        Args:
            request_data: List of input graph nodes in the model, containing the following required fields:
                'data': The data to get predictions on
                'in_tensor_dtype': The datatype of the input, legal types are the keys amongst "dtype_to_number", defined in util.py
                'in_tensor_name': The name of the models input graph node, often 'inputs'

            request_timeout: timeout in seconds

        Returns:
            Empty dict on error, otherwise a dict with keys set as output graph node names and values set to their predicted value

        Example:
            >>> prod_client = ProdClient(host='localhost:9000', model_name='mnist', model_version=1)
            >>> prod_client.predict(
            ...     request_data=[{
            ...         'data': np.asarray([image]),
            ...         'in_tensor_dtype': 'DT_UINT8',
            ...         'in_tensor_name': 'inputs',
            ...     }]
            ... )
        """
        self.logger.info('Sending request to tfserving model')
        self.logger.info('Host: {}'.format(self.host))
        self.logger.info('Model name: {}'.format(self.model_name))
        self.logger.info('Model version: {}'.format(self.model_version))

        # Create gRPC client and request
        t = time.time()
        with grpc.insecure_channel(self.host, options=self.options) as channel:

            self.logger.debug('Establishing insecure channel took: {}'.format(time.time() - t))

            t = time.time()
            stub = PredictionServiceStub(channel)
            self.logger.debug('Creating stub took: {}'.format(time.time() - t))

            t = time.time()
            request = PredictRequest()
            self.logger.debug('Creating request object took: {}'.format(time.time() - t))

            request.model_spec.name = self.model_name

            if self.model_version > 0:
                request.model_spec.version.value = self.model_version

            t = time.time()
            for d in request_data:
                tensor_proto = make_tensor_proto(d['data'], d['in_tensor_dtype'])
                request.inputs[d['in_tensor_name']].CopyFrom(tensor_proto)

            self.logger.debug('Making tensor protos took: {}'.format(time.time() - t))

            try:
                t = time.time()
                predict_response = stub.Predict(request, timeout=request_timeout)

                self.logger.debug('Actual request took: {} seconds'.format(time.time() - t))

                predict_response_dict = predict_response_to_dict(predict_response)

                keys = [k for k in predict_response_dict]
                self.logger.info('Got predict_response with keys: {}'.format(keys))

                return predict_response_dict

            except RpcError as e:
                self.logger.error(e)
                self.logger.error('Prediction failed!')

            return {}
