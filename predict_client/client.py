import tensorflow as tf
import grpc
import logging

from predict_client.predict_pb2 import PredictRequest
from predict_client.prediction_service_pb2 import PredictionServiceStub

logger = logging.getLogger(__name__)


def predict(request_data,  model_name, model_version, host='localhost', port='9000', is_batch_shaped=True, request_timeout=10):
    host = host + ':' + port

    tensor_shape = request_data.shape

    logger.debug('Image shape: ' + str(tensor_shape))

    if is_batch_shaped:
        tensor_shape = (1,) + tensor_shape

    if model_name == 'incv4' or model_name == 'res152':
        features_tensor_proto = tf.contrib.util.make_tensor_proto(request_data, shape=tensor_shape)
    else:
        features_tensor_proto = tf.contrib.util.make_tensor_proto(request_data,
                                                                  dtype=tf.float32, shape=tensor_shape)

    # Create gRPC client and request
    channel = grpc.insecure_channel(host)
    stub = PredictionServiceStub(channel)
    request = PredictRequest()

    request.model_spec.name = model_name

    # request.model_spec.signature_name = 'serving_default'
    if model_version > 0:
        request.model_spec.version.value = model_version

    request.inputs['inputs'].CopyFrom(features_tensor_proto)

    # Send request
    result = stub.Predict(request, timeout=request_timeout)

    return list(result.outputs['scores'].float_val)


if __name__ == '__main__':
    import cv2
    img = cv2.imread('../data/correct-2-2.png', cv2.IMREAD_GRAYSCALE)
    # predict takes an image with shape (height, width, depth)
    img = img.reshape((img.shape + (1,)))
    pred = predict(img, 'mnist', 1)
    print(pred)
