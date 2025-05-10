from dataclasses import asdict, dataclass
from json import JSONEncoder
from typing import Any


@dataclass(frozen=True)
class Prediction:
  x_min: int
  x_max: int
  y_min: int
  y_max: int
  label: str
  confidence: float

  def __post_init__(self) -> None:
    assert 0 <= self.x_min <= self.x_max
    assert 0 <= self.y_min <= self.y_max
    assert self.label != ''
    assert 0 < self.confidence < 1


class PredictionJsonEncoder(JSONEncoder):

  def default(self, o: Any):
    if isinstance(o, Prediction):
      return asdict(o)

    return super().default(o)
