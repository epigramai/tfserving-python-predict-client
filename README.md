# tf_serving_predict_client

## What is this?
The predict client is meant to be used with a model served with tfserving. Because tfserving model server runs a grcp service, it cannot
be requested by just sending a normal HTTP request. The predict_client package is a grcp client that can request the service.
Feel free to use this package to integrate your python apis with tfserving models.

## How to use
def predict(request_data, model_name, model_version, host='localhost', port='9000', is_batch_shaped=True, request_timeout=10):
 - request_data: the data as a numpy array, in batches
 - model_name: the name of the model we want to request
 - model_version: the model version we want to request
 - host: where the tfserving model is hosted
 - port: port of the host
 - is_batch_shaped: by default tfserving expects the images to come in batches, set this parameter to False and the client will add a wrapping dimension before sending the data to the tfserving model
 - request_timeout: timeout sent to the grcp stub
 
 `from predict_client import client`
 `client.predict(request_data, 'mnist', 1, 'localhost', '9000')`
 
## Examples
The mnist example expects an mnist model to be served on localhost:9000. In order to run examples/mnist.py you need to install flask.
Send a POST request to localhost:5000 with an mnist image, and you should get a response with predictions for mnist back.
 
