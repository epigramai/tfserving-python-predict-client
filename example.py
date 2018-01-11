import logging
import numpy as np

from predict_client.prod_client import ProdClient

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

# Make sure you have a model running on localhost:9000
host = 'localhost:9000'
model_name = 'incv4'
model_version = 1

client = ProdClient(host, model_name, model_version, in_tensor_dtype='DT_UINT8')

# Mock up some input data, an image with shape 299,299,3
img = np.zeros((299, 299, 3)).astype(int)

logger.info('Request data shape: ' + str(img.shape))

prediction = client.predict(img, request_timeout=10)

for k in prediction:
    logger.info('Prediction key: ' + str(k) + ', shape: ' + str(prediction[k].shape))

if len(prediction) == 0:
    logger.info('Got empty prediction')
