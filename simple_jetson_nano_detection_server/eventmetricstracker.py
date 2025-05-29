import time
from enum import Enum
from typing import Dict, Generic, TypeVar, Union

from influxdb_client.client.write.point import Point

EventMetricsFields = TypeVar(name='EventMetricsFields', bound=Enum)


class EventMetricsTracker(Generic[EventMetricsFields]):

  def __init__(self) -> None:
    self._values: Dict[EventMetricsFields, int] = {}

  def record(self, field: EventMetricsFields, value: int) -> None:
    self._values[field] = value

  def finalize(self, measurement: str, tags: Dict[str, Union[str, int]] = {}) -> Point:
    assert len(self._values) > 0, 'Nothing to finalize'

    point = Point(measurement).time(time.time_ns())
    for key, value in tags.items():
      point.tag(key, value)

    for key, value in self._values.items():
      point.field(key.name.lower(), value)

    return point
