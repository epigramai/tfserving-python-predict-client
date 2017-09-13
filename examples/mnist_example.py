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
model_name = 'mnist'
model_version = 1

mnist_client = PredictClient(host, model_name, model_version)

img = cv2.imread('2.jpg', 0)
img = np.resize(img, (28, 28, 1))

''' Return value will be None if model not running on host '''
prediction = mnist_client.predict(np.array([img]))

logger.info('Prediction of length: ' + str(len(prediction)))
logger.info('Predictions: ' + str(prediction))
logger.info('Correct class: ' + str(np.argmax(prediction)))
