# tf_serving_predict_client

## What is this?
The predict client is meant to be used with a model served with tfserving. Because tfserving model server runs a grcp service, it cannot
be requested by just sending a normal HTTP request. The predict_client package is a grcp client that can request the service.

Feel free to use this package to integrate your python apis with tfserving models. Or clone the repo and make your own client.



### Install
`pip install git+ssh://git@github.com/epigramai/tfserving_predict_client.git`

## How to use
Assume we have a model server running on localhost:9000, model_name=mnist and model_version=1.
def __init__(self, localhost, envhost, model_name, model_version, num_scores=0):
 - localhost: typically localhost:9000 if you are serving the model locally
 - envhost: if set the PredictClient will look in os.environ[envhost] for a host.
 - model_name: your model name, e.g. 'mnist'
 - model_version: model version, e.g. 1.
 - num_scores: if the prediction fails (let's say the server is not running), then Predict client will return [0] * num_scores where num_scores is the output size of the model that should have been served.
 
def predict(request_data, request_timeout=10):
 - request_data: the data as a numpy array, in batches
 - request_timeout: timeout sent to the grcp stub
 
 `from predict_client.client import PredictClient`
 `client = PredictClient('localhost:9000', None, 'mnist', 1, num_scores=10)`
 `client.predict(request_data, 'mnist', 1, 'localhost', '9000')`
 
 The predict function returns a list of scores. For instance, if you send an mnist image to the client, it will return a list of length 10, where argmax of that list is the correct class.
 
 ## TODO
 - Should be possible to send in batches. If you send multiple images in a batch, then the return value have length batch_size * scores_per_example. E.g if you send 5 mnist images, the return value will be a list with 50 floating points, reshape it and do argmax and you have the classes.
 
## Examples
The mnist example expects an mnist model to be served on localhost:9004. In order to run examples/mnist.py you need to install flask.
Send a POST request to localhost:5000 with an mnist image, and you should get a response with predictions for mnist back.

The concurrent example will run three request in parallel using gevent.
 
