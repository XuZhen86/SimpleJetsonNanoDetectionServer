import time
from enum import Enum
from typing import Dict, Generic, List, TypeVar, Union

from influxdb_client.client.write.point import Point

PerformanceCheckpoints = TypeVar(name='PerformanceCheckpoints', bound=Enum)


class PerformanceTracker(Generic[PerformanceCheckpoints]):

  def __init__(self) -> None:
    self._start_timestamps_ns: Dict[PerformanceCheckpoints, int] = {}
    self._stop_timestamp_ns: Dict[PerformanceCheckpoints, int] = {}
    self._tracked_checkpoints_stack: List[PerformanceCheckpoints] = []

  def __call__(self, checkpoint: PerformanceCheckpoints):
    assert checkpoint not in self._tracked_checkpoints_stack, f'{checkpoint.name} is already being tracked'
    self._tracked_checkpoints_stack.append(checkpoint)
    return self

  def __enter__(self):
    self._start_timestamps_ns[self._tracked_checkpoints_stack[-1]] = time.perf_counter_ns()
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
    checkpoint = self._tracked_checkpoints_stack.pop()
    self._stop_timestamp_ns[checkpoint] = time.perf_counter_ns()

  def start(self, checkpoint: PerformanceCheckpoints) -> None:
    assert checkpoint not in self._tracked_checkpoints_stack, f'{checkpoint.name} is already being tracked with context'
    self._start_timestamps_ns[checkpoint] = time.perf_counter_ns()

  def stop(self, checkpoint: PerformanceCheckpoints) -> None:
    assert checkpoint not in self._tracked_checkpoints_stack, f'{checkpoint.name} should only be stopped with context'
    self._stop_timestamp_ns[checkpoint] = time.perf_counter_ns()

  def finalize(self, measurement: str, tags: Dict[str, Union[str, int]] = {}) -> Point:
    assert len(self._tracked_checkpoints_stack) == 0, (
        f'Cannot finalize before stop tracking {len(self._tracked_checkpoints_stack)} checkpoints')
    assert self._start_timestamps_ns.keys() == self._stop_timestamp_ns.keys(), 'Start/stop calls do not pair'
    assert len(self._start_timestamps_ns) > 0, 'Nothing to finalize'

    point = Point(measurement).time(time.time_ns())
    for key, value in tags.items():
      point.tag(key, value)

    for checkpoint in self._start_timestamps_ns.keys():
      elapsed_ns = self._stop_timestamp_ns[checkpoint] - self._start_timestamps_ns[checkpoint]
      point.field(checkpoint.name.lower() + '_ns', elapsed_ns)

    return point
