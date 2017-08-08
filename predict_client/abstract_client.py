from abc import ABC, abstractmethod


class AbstractPredictClient(ABC):

    def __init__(self, host, model_name, model_version, num_scores=0):
        self.host = host
        self.model_name = model_name
        self.model_version = model_version
        self.num_scores = num_scores

    @abstractmethod
    def predict(self, request_data, request_timeout=10):
        pass
