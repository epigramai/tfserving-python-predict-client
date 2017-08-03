# NB: These two lines are not needed when running a server with gevent workers,
# So don't do it when implementing concurrent in your models!!
# Ask Stian if you have any questions.
from gevent import monkey
monkey.patch_all()

import logging
import cv2
from predict_client.client import PredictClient
from predict_client.util import run_concurrent_requests

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

logger = logging.getLogger(__name__)

incv3_client = PredictClient('localhost:9000', None, 'incv3', 1, num_scores=2048)
incv4_client = PredictClient('localhost:9001', None, 'incv4', 1, num_scores=1536)
res152_client = PredictClient('localhost:9002', None, 'res152', 1, num_scores=2048)

# Open some image
img = cv2.imread('request.png')

incv3_features, incv4_features, res152_features = run_concurrent_requests(img, [incv3_client.predict,
                                                                                incv4_client.predict,
                                                                                res152_client.predict])

logger.info('Num incv3 features:' + str(len(incv3_features)))
logger.info('Num incv4 features:' + str(len(incv4_features)))
logger.info('Num res152 features:' + str(len(res152_features)))

