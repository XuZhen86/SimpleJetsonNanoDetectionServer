import json
import time
from enum import Enum
from typing import Dict, Generic, List, Tuple, TypeVar, Union

from influxdb_client.client.write.point import Point

EventMetricsFields = TypeVar(name='EventMetricsFields', bound=Enum)


class EventMetricsTracker(Generic[EventMetricsFields]):

  def __init__(self) -> None:
    self._values: Dict[Tuple[EventMetricsFields, str], int] = {}

  def record(self, field: EventMetricsFields, value: int, tags: Dict[str, Union[str, int]] = {}) -> None:
    json_tags = json.dumps(tags)
    key = (field, json_tags)
    self._values[key] = value

  def increment(self, field: EventMetricsFields, amount: int = 1, tags: Dict[str, Union[str, int]] = {}) -> None:
    json_tags = json.dumps(tags)
    key = (field, json_tags)
    self._values[key] = self._values.get(key, 0) + amount

  def finalize(self, measurement: str, extra_tags: Dict[str, Union[str, int]] = {}) -> List[Point]:
    assert len(self._values) > 0, 'Nothing to finalize'
    time_ns = time.time_ns()
    points: List[Point] = []

    for (field, json_tags), value in self._values.items():
      tags: Dict[str, Union[str, int]] = json.loads(json_tags)

      point = Point(measurement).field(field.name.lower(), value).time(time_ns)

      for key, value in tags.items():
        point.tag(key, value)
      for key, value in extra_tags.items():
        point.tag(key, value)

      points.append(point)

    return points
