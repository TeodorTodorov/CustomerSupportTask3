from fastapi.testclient import TestClient

from klett_classifier.api import create_app
from klett_classifier.inference import Prediction
from klett_classifier.security import SecuritySettings


class FakeClassifier:
    def predict(self, text):
        return Prediction("Технически проблем", 0.9, {"Технически проблем": 0.9})

    def predict_batch(self, texts):
        return [self.predict(text) for text in texts]


def test_predict_returns_json():
    with TestClient(create_app(FakeClassifier(), security=SecuritySettings(api_key="a" * 32))) as client:
        response = client.post("/v1/predict", headers={"Authorization":"Bearer " + "a" * 32}, json={"text": "Видеото не се отваря"})
        assert response.status_code == 200
        assert response.json()["category"] == "Технически проблем"


def test_rejects_blank_query():
    with TestClient(create_app(FakeClassifier(), security=SecuritySettings(api_key="a" * 32))) as client:
        assert client.post("/v1/predict", headers={"Authorization":"Bearer " + "a" * 32}, json={"text": "   "}).status_code == 422


def test_batch_limits_and_count():
    with TestClient(create_app(FakeClassifier(), security=SecuritySettings(api_key="a" * 32))) as client:
        assert len(client.post("/v1/predict/batch", headers={"Authorization":"Bearer " + "a" * 32}, json={"texts": ["a", "b"]}).json()) == 2
        assert client.post("/v1/predict/batch", headers={"Authorization":"Bearer " + "a" * 32}, json={"texts": ["x"] * 65}).status_code == 422
