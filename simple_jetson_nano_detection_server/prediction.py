from dataclasses import asdict, dataclass
from json import JSONEncoder
from typing import Any

from simple_jetson_nano_detection_server.cocolabel import CocoLabel


@dataclass(frozen=True)
class Prediction:
  x_min: int
  x_max: int
  y_min: int
  y_max: int
  label: CocoLabel
  confidence: float

  def __post_init__(self) -> None:
    assert 0 <= self.x_min <= self.x_max
    assert 0 <= self.y_min <= self.y_max
    assert 0 < self.confidence < 1

  @classmethod
  def build(cls, x_min: int, x_max: int, y_min: int, y_max: int, label: str, confidence: float):
    return cls(x_min, x_max, y_min, y_max, CocoLabel(label), confidence)


class PredictionJsonEncoder(JSONEncoder):

  def default(self, o: Any):
    if isinstance(o, Prediction):
      return asdict(o)

    return super().default(o)
