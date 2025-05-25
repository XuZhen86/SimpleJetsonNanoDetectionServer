import json

from absl.testing import parameterized

from simple_jetson_nano_detection_server.prediction import Prediction, PredictionJsonEncoder


class TestPrediction(parameterized.TestCase):

  @parameterized.parameters(
      (-1, 0, 0, 0, 'label', 0.5),
      (0, -1, 0, 0, 'label', 0.5),
      (0, 0, -1, 0, 'label', 0.5),
      (0, 0, 0, -1, 'label', 0.5),
      (0, 0, 0, 0, '', 0.5),
      (0, 0, 0, 0, 'label', 0.0),
      (0, 0, 0, 0, 'label', 1.0),
      (1, 0, 0, 0, 'label', 0.5),
      (0, 0, 1, 0, 'label', 0.5),
  )
  def test_invalidValues(self, x_min: int, x_max: int, y_min: int, y_max: int, label: str, confidence: float):
    with self.assertRaises(Exception):
      Prediction(x_min, x_max, y_min, y_max, label, confidence)

  @parameterized.parameters(
      (0, 0, 0, 0, 'label', 0.5),
      (0, 1, 0, 0, 'label', 0.5),
      (0, 0, 0, 1, 'label', 0.5),
      (0, 0, 0, 0, 'label', 0.01),
      (0, 0, 0, 0, 'label', 0.99),
  )
  def test_validValues(self, x_min: int, x_max: int, y_min: int, y_max: int, label: str, confidence: float):
    Prediction(x_min, x_max, y_min, y_max, label, confidence)


class TestPredictionJsonEncoder(parameterized.TestCase):

  @parameterized.parameters(
      (
          Prediction(0, 0, 0, 0, 'label', 0.5),
          '{"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0, "label": "label", "confidence": 0.5}',
      ),
      (
          Prediction(0, 1, 0, 1, 'label', 0.9),
          '{"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "label": "label", "confidence": 0.9}',
      ),
  )
  def test_convertsToJson(self, p: Prediction, j: str):
    self.assertJsonEqual(json.dumps(p, cls=PredictionJsonEncoder), j)
