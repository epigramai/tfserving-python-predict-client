import logging
import os
import time
import cv2
import numpy as np

# from predict_client.mock_client import MockClient
from predict_client.prod_client import ProdClient

# Logger initialization
# This must happen before any calls to debug(), info(), etc.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

host = 'localhost:9000'
model_name = 'incv3'
model_version = 1

# Choose between inmemory og hosted
client = ProdClient(host, model_name, model_version, in_tensor_dtype='float32')
# client = MockClient(os.path.join(os.path.dirname(__file__), '../models/incv3_2048/1/'))

base_path = os.path.join(os.path.dirname(__file__), '../test_data/catdogs')
times = 0
i = 0

for f in os.listdir(base_path):
    if f == '.DS_Store':
        continue

    i += 1

    logger.info('Req file name: ' + str(f))

    img = cv2.imread(os.path.join(base_path, f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))

    img = img / 255
    img -= 0.5
    img *= 2
    img_batch = np.array(img)

    logger.info('Req data shape: ' + str(img_batch.shape))

    t = time.time()
    prediction = client.predict(img_batch, request_timeout=10)

    req_time = time.time() - t
    logger.info('Request time: ' + str(req_time))

    times += req_time

    for k in prediction:
        logger.info('Prediction key: ' + str(k) + ', shape: ' + str(prediction[k].shape))

    if len(prediction) == 0:
        logger.info('Got empty prediction')

logger.info('Avg. req time: ' + str(times / i))
