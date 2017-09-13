# Python, flask and various api code imports
import logging
import cv2
import numpy as np

from predict_client.prod_client import PredictClient


# Logger initialization
# This must happen before any calls to debug(), info(), etc.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

host = 'localhost:9000'
model_name = 'incv4'
model_version = 1

incv4_client = PredictClient(host, model_name, model_version)

img = cv2.imread('test_image.jpg')

''' Return value will be None if model not running on host '''
prediction = incv4_client.predict(img)

logger.info('Features of length: ' + str(len(prediction)))
logger.info('First 10 features: ' + str(prediction[:10]))
