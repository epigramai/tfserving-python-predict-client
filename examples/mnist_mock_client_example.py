# Python, flask and various api code imports
from tfwrapper import twimage
from predict_client.mock_client import MockPredictClient


host = 'localhost:9004'
model_name = 'mnist'
model_version = 1

mnist_mock_client = MockPredictClient(host,
                                      model_name,
                                      model_version,
                                      '/Users/slp/.tfwrapper/serving_models/incv4_1536/1/')


img = twimage.imread('request.png')

#mnist_mock_client.predict(img)
