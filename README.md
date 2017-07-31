# tf_serving_predict_client

## What is this?
The predict client is meant to be used with a model served with tfserving. Because tfserving model server runs a grcp service, it cannot
be requested by just sending a normal HTTP request. The predict_client package is a grcp client that can request the service.
Feel free to use this package to integrate your python apis with tfserving models.

## Examples
The mnist example expects an mnist model to be served on localhost:9000. In order to run examples/mnist.py you need to install flask.
Send a POST request to localhost:5000 with an mnist image, and you should get a response with predictions for mnist back.
 