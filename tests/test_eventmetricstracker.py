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

    points = self.tracker.finalize('m')
    self.assertListEqual([p.to_line_protocol() for p in points], [
        'm field_1=111i 1700000000000000000',
        'm field_2=222i 1700000000000000000',
    ])

  def test_incrementAndFinalize(self):
    self.tracker.increment(_EventMetricsField.FIELD_1, 11)
    self.tracker.increment(_EventMetricsField.FIELD_2)

    points = self.tracker.finalize('m')
    self.assertListEqual([p.to_line_protocol() for p in points], [
        'm field_1=11i 1700000000000000000',
        'm field_2=1i 1700000000000000000',
    ])

  def test_recordUsesNewValues(self):
    self.tracker.record(_EventMetricsField.FIELD_1, 111)
    self.tracker.record(_EventMetricsField.FIELD_1, 222)

    points = self.tracker.finalize('m')
    self.assertListEqual([p.to_line_protocol() for p in points], [
        'm field_1=222i 1700000000000000000',
    ])

  def test_recordOverridesIncrementedValues(self):
    self.tracker.increment(_EventMetricsField.FIELD_1, 100)
    self.tracker.record(_EventMetricsField.FIELD_1, 111)

    points = self.tracker.finalize('m')
    self.assertListEqual([p.to_line_protocol() for p in points], [
        'm field_1=111i 1700000000000000000',
    ])

  def test_incrementsRecordedValues(self):
    self.tracker.record(_EventMetricsField.FIELD_1, 111)
    self.tracker.increment(_EventMetricsField.FIELD_1, 1)

    points = self.tracker.finalize('m')
    self.assertListEqual([p.to_line_protocol() for p in points], [
        'm field_1=112i 1700000000000000000',
    ])

  def test_finalizeWithExtraTags(self):
    self.tracker.record(_EventMetricsField.FIELD_1, 111, {'tag4': 4})

    points = self.tracker.finalize('m', {'tag1': 'value1', 'tag2': 'value2', 'tag3': 3})
    self.assertListEqual([p.to_line_protocol() for p in points], [
        'm,tag1=value1,tag2=value2,tag3=3,tag4=4 field_1=111i 1700000000000000000',
    ])

  def test_differentFieldSameTags(self):
    self.tracker.record(_EventMetricsField.FIELD_1, 111, {'tag1': 'value1'})
    self.tracker.increment(_EventMetricsField.FIELD_2, 222, {'tag1': 'value1'})

    points = self.tracker.finalize('m')
    self.assertListEqual([p.to_line_protocol() for p in points], [
        'm,tag1=value1 field_1=111i 1700000000000000000',
        'm,tag1=value1 field_2=222i 1700000000000000000',
    ])

  def test_sameFieldDifferentTags(self):
    self.tracker.record(_EventMetricsField.FIELD_1, 111, {'tag1': 'value1'})
    self.tracker.increment(_EventMetricsField.FIELD_1, 222, {'tag1': 'value2'})

    points = self.tracker.finalize('m')
    self.assertListEqual([p.to_line_protocol() for p in points], [
        'm,tag1=value1 field_1=111i 1700000000000000000',
        'm,tag1=value2 field_1=222i 1700000000000000000',
    ])
