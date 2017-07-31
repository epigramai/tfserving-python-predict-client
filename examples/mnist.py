# Python, flask and various api code imports
import logging
from flask import Flask, request, Response, jsonify
import socket
import cv2

from predict_client import client

# Logger initialization
# This must happen before any calls to debug(), info(), etc.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

# API initialization
app = Flask(__name__)

MODEL_VERSION = 1


@app.route('/predict', methods=['POST'])
def predict():
    logger.info('/predict, hostname: ' + str(socket.gethostname()))

    if 'image' not in request.files:
        logger.info('Missing image parameter')
        return Response('Missing image parameter', 400)

    # Write image to disk
    with open('request.png', 'wb') as f:
        f.write(request.files['image'].read())

    img = cv2.imread('request.png', 0)
    prediction = client.predict(img.reshape((img.shape + (1,))), 'mnist', MODEL_VERSION,
                                host='localhost', port='9000')

    logger.info('Prediction of length:' + str(len(prediction)))

    ''' Convert the dict to json and return response '''
    return jsonify(
        prediction=prediction,
        prediction_length=len(prediction),
        hostname=str(socket.gethostname())
    )


@app.errorhandler(500)
def server_error(e):
    logger.error(str(e))
    response = Response('An internal error occurred. ' + str(e), 500)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')
