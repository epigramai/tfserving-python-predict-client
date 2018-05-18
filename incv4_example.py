import logging
import numpy as np

from predict_client.prod_client import ProdClient

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

# Make sure you have a model running on localhost:9000
host = 'localhost:9001'
model_name = 'incv4'
model_version = 1
img = np.zeros((299, 299, 3)).astype(int)
req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_UINT8', 'data': img}]

client = ProdClient(host, model_name, model_version)

prediction = client.predict(req_data, request_timeout=10)
for k in prediction:
    logger.info('Prediction key: {}, shape: {}'.format(k, prediction[k].shape))
