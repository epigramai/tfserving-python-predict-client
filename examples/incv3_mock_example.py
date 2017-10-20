# Python, flask and various api code imports
import logging
import cv2
import numpy as np
from predict_client.prod_client import PredictClient
from predict_client.mock_client import MockPredictClient
import time
import os

# Logger initialization
# This must happen before any calls to debug(), info(), etc.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

host = 'localhost:9000'
model_name = 'incv3'
model_version = 1

# incv3_client = PredictClient(host, model_name, model_version)
incv3_client = MockPredictClient(os.path.join(os.path.dirname(__file__), '../models/incv3_bottleneck/1'))

base_path = os.path.join(os.path.dirname(__file__), '../test_data/images')
times = 0
i = 0
for f in os.listdir(base_path):
    if f == '.DS_Store':
        continue

    i += 1

    img = cv2.imread(os.path.join(base_path, f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))

    t = time.time()
    prediction = incv3_client.predict(np.array([img]))

    req_time = time.time() - t

    logger.info('Request time: ' + str(req_time))
    times += req_time

    logger.info('Features of length: ' + str(len(prediction)))
    logger.info('First 10 features: ' + str(prediction[:10]))

logger.info('Avg. req time: ' + str(times / i))
