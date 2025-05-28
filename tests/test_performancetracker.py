import time
from enum import Enum, auto
from unittest.mock import Mock, patch

from absl.testing import parameterized

from simple_jetson_nano_detection_server.performancetracker import PerformanceTracker


class _PerformanceCheckpoint(Enum):
  CHECKPOINT_1 = auto()
  CHECKPOINT_2 = auto()


@patch.object(time, time.time_ns.__name__, Mock(return_value=1700000000000000000))
class TestPerformanceTracker(parameterized.TestCase):

  def setUp(self):
    self.tracker: PerformanceTracker[_PerformanceCheckpoint] = PerformanceTracker()
    return super().setUp()

  @patch.object(time, time.perf_counter_ns.__name__, Mock(side_effect=[69, 420]))
  def test_trackingWithContextManager(self):
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      pass

    point = self.tracker.finalize('m')
    self.assertEqual(point.to_line_protocol(), 'm checkpoint_1_ns=351i 1700000000000000000')

  @patch.object(time, time.perf_counter_ns.__name__, Mock(side_effect=[69, 100, 420, 1000]))
  def test_trackingWithNestedContextManager(self):
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      with self.tracker(_PerformanceCheckpoint.CHECKPOINT_2):
        pass

    point = self.tracker.finalize('m')
    self.assertEqual(point.to_line_protocol(), 'm checkpoint_1_ns=931i,checkpoint_2_ns=320i 1700000000000000000')

  def test_contextManager_alreadyTracking_raises(self):
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      with self.assertRaisesWithLiteralMatch(Exception, 'CHECKPOINT_1 is already being tracked'):
        with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
          pass

  @patch.object(time, time.perf_counter_ns.__name__, Mock(side_effect=[69, 100, 420, 1000]))
  def test_contextManager_allowsNewTrackingWithTheSameCheckpoing(self):
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      pass
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      pass

    point = self.tracker.finalize('m')
    self.assertEqual(point.to_line_protocol(), 'm checkpoint_1_ns=580i 1700000000000000000')

  def test_start_alreadyTrackedByContextManager_raises(self):
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      with self.assertRaisesWithLiteralMatch(Exception, 'CHECKPOINT_1 is already being tracked with context'):
        self.tracker.start(_PerformanceCheckpoint.CHECKPOINT_1)

  def test_stop_alreadyTrackedByContextManager_raises(self):
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      with self.assertRaisesWithLiteralMatch(Exception, 'CHECKPOINT_1 should only be stopped with context'):
        self.tracker.stop(_PerformanceCheckpoint.CHECKPOINT_1)

  @patch.object(time, time.perf_counter_ns.__name__, Mock(side_effect=[69, 420]))
  def test_trackingWithStartStop(self):
    self.tracker.start(_PerformanceCheckpoint.CHECKPOINT_1)
    self.tracker.stop(_PerformanceCheckpoint.CHECKPOINT_1)

    point = self.tracker.finalize('m')
    self.assertEqual(point.to_line_protocol(), 'm checkpoint_1_ns=351i 1700000000000000000')

  def test_finalizeWhileTracking_raises(self):
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      with self.tracker(_PerformanceCheckpoint.CHECKPOINT_2):
        with self.assertRaisesWithLiteralMatch(Exception, 'Cannot finalize before stop tracking 2 checkpoints'):
          self.tracker.finalize('m')

  def test_finalizeWithMissedStart_raises(self):
    self.tracker.start(_PerformanceCheckpoint.CHECKPOINT_1)
    self.tracker.stop(_PerformanceCheckpoint.CHECKPOINT_1)
    self.tracker.stop(_PerformanceCheckpoint.CHECKPOINT_2)

    with self.assertRaisesWithLiteralMatch(Exception, 'Start/stop calls do not pair'):
      self.tracker.finalize('m')

  def test_finalizeWithMissedStop_raises(self):
    self.tracker.start(_PerformanceCheckpoint.CHECKPOINT_1)
    self.tracker.stop(_PerformanceCheckpoint.CHECKPOINT_1)
    self.tracker.start(_PerformanceCheckpoint.CHECKPOINT_2)

    with self.assertRaisesWithLiteralMatch(Exception, 'Start/stop calls do not pair'):
      self.tracker.finalize('m')

  def test_finalizeWithEmptyTracker_raises(self):
    with self.assertRaisesWithLiteralMatch(Exception, 'Nothing to finalize'):
      self.tracker.finalize('m')

  @patch.object(time, time.perf_counter_ns.__name__, Mock(side_effect=[69, 420]))
  def test_finalizeWithTags(self):
    with self.tracker(_PerformanceCheckpoint.CHECKPOINT_1):
      pass

    point = self.tracker.finalize('m', {'tag1': 'value1', 'tag2': 'value2', 'tag3': 3})
    self.assertEqual(point.to_line_protocol(),
                     'm,tag1=value1,tag2=value2,tag3=3 checkpoint_1_ns=351i 1700000000000000000')
