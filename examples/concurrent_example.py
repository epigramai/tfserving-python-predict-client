# NB: These two lines are not needed when running a server with gevent workers,
# So don't do it when implementing it.
# Ask Stian if you have any questions.
from gevent import monkey

monkey.patch_all()

import os
import logging
import cv2
import numpy as np
from predict_client.prod_client import ProdClient
from predict_client.util import run_concurrent_requests

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

logger = logging.getLogger(__name__)

# Do this with PredictClient to see behavior if the model server is down
incv3_client = ProdClient('localhost:9000', 'incv3', 1, in_tensor_dtype='float32')
incv4_client = ProdClient('localhost:9001', 'incv4', 1, in_tensor_dtype='uint8')

# Open some image
img = cv2.imread(os.path.join(os.path.dirname(__file__), '../test_data/catdogs/dog.2347.jpg'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (299, 299))

img_incv3 = img / 255
img_incv3 -= 0.5
img_incv3 *= 2
img_incv3 = np.array(img_incv3)

img_incv4 = np.array(img)

print(img_incv3.shape)
print(img_incv4.shape)

incv3_features, incv4_features = run_concurrent_requests([img_incv3, img_incv4], [incv3_client.predict,
                                                                                  incv4_client.predict])

print(incv3_features)
print(incv4_features)
