# tfserving_predict_client

## What is this?
The predict client is meant to be used with a model served with tfserving. Because tfserving model server runs a grcp service, it cannot
be requested by just sending a normal HTTP request. The predict_client package is a grcp client that can request the service.

Feel free to use this package to integrate your python apis with tfserving models. Or clone the repo and make your own client.

Thanks to https://github.com/tobegit3hub/tensorflow_template_application for working grcp pb-files and inspiration :)

### Install
`pip install git+ssh://git@github.com/epigramai/tfserving_predict_client.git`

### If you need a model server
There is one here https://hub.docker.com/r/epigramai/model-server/

## How to use
Assume we have a model server running on localhost:9000, model_name=mnist and model_version=1.

#### predict_client.prod_client.PredictClient
def __init__(self, host, model_name, model_version):
 - host: the host (e.g. localhost:9000)
 - model_name: your model name, e.g. 'mnist'
 - model_version: model version, e.g. 1.
 
PredictClient.predict(self, request_data, request_timeout=10):
 - request_data: the data as a numpy array, in batches
 - request_timeout: timeout sent to the grcp stub
 
 `from predict_client.prod_client import PredictClient`
 
 `client = PredictClient('localhost:9000', 'mnist', 1)`
 
 `client.predict(request_data)`
 
 The predict function returns a list of scores. For instance, if you send an mnist image to the client, it will return a list of length 10, where argmax of that list is the correct class.
 The predict client will return None if it fails.
 
#### predict_client.prod_client.MockPredictClient
def __init__(self, host, model_name, model_version):
 - host, model_name and model_version same as PredictClient
 - num_scores: if the prediction fails (let's say the server is not running), then Predict client will return [0] * num_scores where num_scores is the output size of the model that should have been served.
 
MockPredictClient.predict(self, request_data, request_timeout=10):
 - request_data and request_timeout same as PredictClient
 
 `from predict_client.mock_client import MockPredictClient`
 
 `client = MockPredictClient('localhost:9000', 'mnist', 1, num_scores=10)`
 
 `client.predict(request_data)`
 
 The predict function returns a list of scores. For instance, if you send an mnist image to the client, it will return a list of length 10, where argmax of that list is the correct class.
 The mock predict client will return \[0\] * num_scores if it fails. 
 
 ## TODO
 - Should be possible to send in batches. If you send multiple images in a batch, then the return value have length batch_size * scores_per_example. E.g if you send 5 mnist images, the return value will be a list with 50 floating points, reshape it and do argmax and you have the classes.
 
## Examples
The mnist example expects an mnist model to be served on localhost:9004. In order to run examples/mnist.py you need to install flask.
Send a POST request to localhost:5000 with an mnist image, and you should get a response with predictions for mnist back.

The concurrent example will run three request in parallel using gevent.
 
