# tf_serving_predict_client

## What is this?
The predict client is meant to be used with a model served with tfserving. Because tfserving model server runs a grcp service, it cannot
be requested by just sending a normal HTTP request. The predict_client package is a grcp client that can request the service.
Feel free to use this package to integrate your python apis with tfserving models.

### Install
`pip install git+ssh://git@github.com/epigramai/tfserving_predict_client.git`

## How to use
def predict(request_data, model_name, model_version, host='localhost:9000', is_batch_shaped=True, request_timeout=10):
 - request_data: the data as a numpy array, in batches
 - model_name: the name of the model we want to request
 - model_version: the model version we want to request
 - host: where the tfserving model is hosted
 - is_batch_shaped: by default tfserving expects the images to come in batches, set this parameter to False and the client will add a wrapping dimension before sending the data to the tfserving model
 - request_timeout: timeout sent to the grcp stub
 
 `from predict_client import client`
 
 `client.predict(request_data, 'mnist', 1, 'localhost', '9000')`
 
 The predict function returns a list of scores. For instance, if you send an mnist image to the client, it will return a list of length 10, where argmax of that list is the correct class. If you send multiple images in a batch, then the return value have length batch_size * scores_per_example. E.g if you send 5 mnist images, the return value will be a list with 50 floating points, reshape it and do argmax and you have the classes.
 
## Examples
The mnist example expects an mnist model to be served on localhost:9000. In order to run examples/mnist.py you need to install flask.
Send a POST request to localhost:5000 with an mnist image, and you should get a response with predictions for mnist back.
 
