import time
from enum import Enum, auto
from unittest.mock import Mock, patch

from absl.testing import parameterized

from simple_jetson_nano_detection_server.eventmetricstracker import EventMetricsTracker


class _EventMetricsField(Enum):
  FIELD_1 = auto()
  FIELD_2 = auto()


@patch.object(time, time.time_ns.__name__, Mock(return_value=1700000000000000000))
class TestEventMetricsTracker(parameterized.TestCase):

  def setUp(self):
    self.tracker: EventMetricsTracker[_EventMetricsField] = EventMetricsTracker()
    return super().setUp()

  def test_recordAndFinalize(self):
    self.tracker.record(_EventMetricsField.FIELD_1, 111)
    self.tracker.record(_EventMetricsField.FIELD_2, 222)

    point = self.tracker.finalize('m')
    self.assertEqual(point.to_line_protocol(), 'm field_1=111i,field_2=222i 1700000000000000000')

  def test_recordUsesNewValues(self):
    self.tracker.record(_EventMetricsField.FIELD_1, 111)
    self.tracker.record(_EventMetricsField.FIELD_1, 222)

    point = self.tracker.finalize('m')
    self.assertEqual(point.to_line_protocol(), 'm field_1=222i 1700000000000000000')

  def test_finalizeWithTags(self):
    self.tracker.record(_EventMetricsField.FIELD_1, 111)

    point = self.tracker.finalize('m', {'tag1': 'value1', 'tag2': 'value2', 'tag3': 3})
    self.assertEqual(point.to_line_protocol(), 'm,tag1=value1,tag2=value2,tag3=3 field_1=111i 1700000000000000000')
