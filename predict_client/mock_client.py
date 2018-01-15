class MockClient:
    def __init__(self, mock_reponse):
        self.mock_reponse = mock_reponse

    def predict(self, request_data, **kwargs):
        return self.mock_reponse
