import logging
import gevent
import numpy as np

logger = logging.getLogger(__name__)


def run_concurrent_requests(request_data, clients):
    """ Makes predictions from all clients concurrently.

        Arguments:
        request_data -- data that can be fed into all clients
        clients -- a list of PredictClient.predict functions

        Returns:
        A list of predictions from each client.
    """

    jobs = [gevent.spawn(c, d) for c, d in zip(clients, request_data)]
    gevent.joinall(jobs, timeout=10)

    for j in jobs:
        print(j)

    return list(map(lambda j: j.value, jobs))


def result_to_dict(result):
    dtype_map = {
        7: 'string_val',
        1: 'float_val'
    }

    result_dict = dict()

    for k in result.outputs:
        shape = [x.size for x in result.outputs[k].tensor_shape.dim]

        logger.debug('Key: ' + k + ', shape: ' + str(shape))

        dtype_constant = result.outputs[k].dtype

        if dtype_constant not in dtype_map:
            logger.error('Tensor output data type not supported. Returning empty dict.')
            result_dict[k] = 'value not found'

        result_dict[k] = np.array(eval('result.outputs[k].' + dtype_map[dtype_constant])).reshape(shape)

    return result_dict
