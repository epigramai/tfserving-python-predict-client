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
Check out the examples. The download scripts with test data and models will not work unless you have access to Epigram AI GCS buckets, but feel free to
use the examples as a starting point.

Although the models and test data is hidden to the public, the predict client is open source.

### predict_client.prod_client ProdClient
def __init__(self, host, model_name, model_version):
 - host: the host (e.g. localhost:9000)
 - model_name: your model name, e.g. 'mnist'
 - model_version: model version, e.g. 1.
 
ProdClient.predict(self, request_data, request_timeout=10):
 - request_data: the data as a numpy array, with the same shape as the model's input tensor
 - request_timeout: timeout sent to the grcp stub
 
 `from predict_client.prod_client import ProdClient`
 
 `client = PredictClient('localhost:9000', 'mnist', 1)`
 
 `client.predict(request_data)`
 
 The predict function returns a dictionary with keys and values for each output tensor. The values in the dictionary will have the same shapes as
 the output tensor's shape. If an error occurs, predict will return an empty dict.
 
### predict_client.inmemory_client InMemoryClient
def __init__(self, model_path):
 - model_path
 
InMemoryClient.predict(self, request_data, request_timeout=None):
 - request_data and request_timeout same as ProdClient, except request_timeout not used in mock client.
 
 `from predict_client.inmemory_client import InMemoryClient`
 
 `client = InMemoryClient('path/to/model.pb')`
 
 `client.predict(request_data)`
 
The predict function returns a dictionary with keys and values for each output tensor. The values in the dictionary will have the same shapes as
the output tensor's shape. If an error occurs, predict will return an empty dict.
  
### predict_client.mock_client MockClient
def __init__(self, mock_response):
 - mock_response
 
MockClient.predict(self, request_data, request_timeout=None):
 - request_data and request_timeout same as ProdClient, except request_timeout not used in mock client.
 
 `from predict_client.mock_client import MockClient`
 
 `A mock response: mock_response={'scores': np.array([0.998, 0.002813, 0.0283]), 'classes': np.array(['dog', 'cat', 'horse'])}`
 
 `client = MockClient(mock_response)`
 
 `client.predict(request_data)`
 
The mock client predict function simply returns the mock response. 