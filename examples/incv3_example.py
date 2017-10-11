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
model_version = 2

incv3_client = PredictClient(host, model_name, model_version)

# for f in os.listdir('../test_data/res'):
#     if not f.endswith('.jpg'):
#         continue

img = cv2.imread(
    os.path.join('/Users/slp/Development/tfserving_predict_client/test_data/car_images/Image1857451_5.jpg'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (299, 299))
# img = np.expand_dims(img, axis=0)

t = time.time()

''' Return value will be None if model not running on host '''
prediction = incv3_client.predict(np.array([img]))

logger.info('Request time: ' + str(time.time() - t))

logger.info('Features of length: ' + str(len(prediction)))
logger.info('First 10 features: ' + str(prediction[:10]))
# time.sleep(20)
