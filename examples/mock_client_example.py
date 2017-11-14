import logging
import os
import cv2
import numpy as np

from predict_client.mock_client import MockClient

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

logger = logging.getLogger(__name__)

client = MockClient({'scores': np.array([0.998, 0.002813, 0.0283]), 'classes': np.array(['dog', 'cat', 'horse'])})

base_path = os.path.join(os.path.dirname(__file__), '../test_data/catdogs')
times = 0
i = 0


img = cv2.imread(os.path.join(base_path, 'dog.2347.jpg'))


prediction = client.predict(img, request_timeout=10)

logger.info('Prediction key: ' + str(k) + ', shape: ' + str(prediction[k].shape))


