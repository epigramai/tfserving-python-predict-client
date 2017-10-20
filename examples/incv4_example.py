# Python, flask and various api code imports
import logging
import cv2
import numpy as np
from predict_client.prod_client import PredictClient
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

incv3_client = PredictClient(host, model_name, model_version)

base_path = os.path.join(os.path.dirname(__file__), '../test_data/images')
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
    img_batch = np.array([img])

    logger.info('Req data shape: ' + str(img_batch.shape))

    t = time.time()
    prediction = incv3_client.predict(img_batch)

    req_time = time.time() - t
    logger.info('Request time: ' + str(req_time))

    times += req_time

    logger.info('Got features with shape: ' + str(prediction.shape))

logger.info('Avg. req time: ' + str(times / i))
