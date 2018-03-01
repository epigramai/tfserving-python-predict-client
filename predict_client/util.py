import logging
import numpy as np
from predict_client.pbs.tensor_pb2 import TensorProto
from predict_client.dict_to_protobuf import dict_to_protobuf

logger = logging.getLogger(__name__)

# Should be all values from protos/types.proto
dtype_to_number = {
    'DT_INVALID': 0,
    'DT_FLOAT': 1,
    'DT_DOUBLE': 2,
    'DT_INT32': 3,
    'DT_UINT8': 4,
    'DT_INT16': 5,
    'DT_INT8': 6,
    'DT_STRING': 7,
    'DT_COMPLEX64': 8,
    'DT_INT64': 9,
    'DT_BOOL': 10,
    'DT_QINT8': 11,
    'DT_QUINT8': 12,
    'DT_QINT32': 13,
    'DT_BFLOAT16': 14,
    'DT_QINT16': 15,
    'DT_QUINT16': 16,
    'DT_UINT16': 17,
    'DT_COMPLEX128': 18,
    'DT_HALF': 19,
    'DT_RESOURCE': 20
}

number_to_dtype_value = {
    1: 'float_val',
    2: 'double_val',
    3: 'int_val',
    4: 'int_val',
    5: 'int_val',
    6: 'int_val',
    7: 'string_val',
    8: 'scomplex_val',
    9: 'int64_val',
    10: 'bool_val',
    18: 'dcomplex_val',
    19: 'half_val',
    20: 'resource_handle_val'
}


def predict_response_to_dict(predict_response):
    predict_response_dict = dict()

    for k in predict_response.outputs:
        shape = [x.size for x in predict_response.outputs[k].tensor_shape.dim]

        logger.debug('Key: ' + k + ', shape: ' + str(shape))

        dtype_constant = predict_response.outputs[k].dtype

        if dtype_constant not in number_to_dtype_value:
            logger.error('Tensor output data type not supported. Returning empty dict.')
            predict_response_dict[k] = 'value not found'

        if shape == [1]:
            predict_response_dict[k] = eval('predict_response.outputs[k].' + number_to_dtype_value[dtype_constant])[0]
        else:
            predict_response_dict[k] = np.array(
                eval('predict_response.outputs[k].' + number_to_dtype_value[dtype_constant])).reshape(shape)

    return predict_response_dict


def make_tensor_proto(data, dtype):
    tensor_proto = TensorProto()

    if type(dtype) is str:
        dtype = dtype_to_number[dtype]

    dim = [{'size': 1}]
    values = [data]

    if hasattr(data, 'shape'):
        dim = [{'size': dim} for dim in data.shape]
        values = list(data.reshape(-1))

    tensor_proto_dict = {
        'dtype': dtype,
        'tensor_shape': {
            'dim': dim
        },
        number_to_dtype_value[dtype]: values
    }

    dict_to_protobuf(tensor_proto_dict, tensor_proto)

    return tensor_proto
