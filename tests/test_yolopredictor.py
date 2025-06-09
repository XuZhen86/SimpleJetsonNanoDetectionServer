import time
from itertools import chain
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from absl.testing import flagsaver, parameterized
from line_protocol_cache.lineprotocolcache import LineProtocolCache

from simple_jetson_nano_detection_server.prediction import Prediction
from simple_jetson_nano_detection_server.yolopredictor import _HALF_PRECISION, _IMAGE_SIZE, YoloPredictor

LINE_PROTOCOL_CACHE_PUT = Mock(return_value=None)


@patch.object(LineProtocolCache, LineProtocolCache.put.__name__, LINE_PROTOCOL_CACHE_PUT)
@patch.object(time, time.time_ns.__name__, Mock(return_value=1700000000000000000))
class TestYoloPredictor(parameterized.TestCase):

  def setUp(self):
    self.saved_flags = flagsaver.as_parsed(
        (_HALF_PRECISION, str(False)),
        (_IMAGE_SIZE, str(12345)),
    )
    self.saved_flags.__enter__()

    mock_xyxy = Mock(tolist=Mock(return_value=[
        [132.5, 104.5, 177.5, 141.875],
        [264.25, 173.40625, 319.75, 179.09375],
        [111.5, 164.5625, 319.5, 319.4375],
        [111.5, 164.5625, 319.5, 319.4375],
        [111.5, 164.5625, 319.5, 319.4375],
    ]))
    mock_conf = Mock(tolist=Mock(return_value=[
        0.6460136771202087,
        0.42441198229789734,
        0.29746994376182556,
        0.29746994376182556,
        0.40346994376182556,
    ]))
    mock_cls = Mock(tolist=Mock(return_value=[
        1.0,
        2.0,
        3.0,
        3.0,
        3.0,
    ]))
    mock_boxes = Mock(xyxy=mock_xyxy, conf=mock_conf, cls=mock_cls)
    mock_result = Mock(boxes=mock_boxes)
    mock_results = [mock_result]

    self.mock_yolo_predict = Mock(return_value=mock_results)
    self.mock_yolo = Mock(predict=self.mock_yolo_predict, names={1: 'person', 2: 'bicycle', 3: 'car'})

    YoloPredictor.set_model(self.mock_yolo)

    LINE_PROTOCOL_CACHE_PUT.reset_mock(return_value=True, side_effect=True)

    return super().setUp()

  def tearDown(self) -> None:
    self.saved_flags.__exit__(None, None, None)
    return super().tearDown()

  def _assert_line_protocols(self, expected: List[str]) -> None:
    points = chain.from_iterable([call_arg.args[0] for call_arg in LINE_PROTOCOL_CACHE_PUT.call_args_list])
    line_protocols = [p.to_line_protocol() for p in points]
    self.assertListEqual(line_protocols, expected)

  def _assertDictContainsSubset(self, subset: Dict[Any, Any], dictionary: Dict[Any, Any], msg: object = None) -> None:
    self.assertEqual(dictionary, {**dictionary, **subset}, msg)

  def test_noModel_raises(self):
    YoloPredictor._model = None

    with self.assertRaisesWithLiteralMatch(Exception, 'A model must be set before prediction'):
      YoloPredictor.predict(b'image-bytes')
    LINE_PROTOCOL_CACHE_PUT.assert_not_called()

  def test_noResults_raises(self):
    self.mock_yolo_predict.return_value = []

    with self.assertRaisesWithLiteralMatch(Exception, 'There must be exactly 1 result, got 0 instead'):
      YoloPredictor.predict(b'image-bytes')

    self._assert_line_protocols([
        'prediction_input image_bytes=11i 1700000000000000000',
    ])

  def test_moreThan1Results_raises(self):
    self.mock_yolo_predict.return_value = [Mock(), Mock()]

    with self.assertRaisesWithLiteralMatch(Exception, 'There must be exactly 1 result, got 2 instead'):
      YoloPredictor.predict(b'image-bytes')

    self._assert_line_protocols([
        'prediction_input image_bytes=11i 1700000000000000000',
    ])

  def test_boxesIsNone_raises(self):
    self.mock_yolo_predict.return_value = [Mock(boxes=None)]

    with self.assertRaisesWithLiteralMatch(Exception, 'Boxes cannot be None'):
      YoloPredictor.predict(b'image-bytes')

    self._assert_line_protocols([
        'prediction_input image_bytes=11i 1700000000000000000',
    ])

  def test_convertsToPredictions(self):
    predictions = YoloPredictor.predict(b'image-bytes')

    self.assertEqual(predictions, [
        Prediction(x_min=132, x_max=177, y_min=104, y_max=141, label='person', confidence=0.6460136771202087),
        Prediction(x_min=264, x_max=319, y_min=173, y_max=179, label='bicycle', confidence=0.42441198229789734),
        Prediction(x_min=111, x_max=319, y_min=164, y_max=319, label='car', confidence=0.29746994376182556),
        Prediction(x_min=111, x_max=319, y_min=164, y_max=319, label='car', confidence=0.29746994376182556),
        Prediction(x_min=111, x_max=319, y_min=164, y_max=319, label='car', confidence=0.40346994376182556),
    ])
    self._assert_line_protocols([
        'prediction_input image_bytes=11i 1700000000000000000',
        'prediction_output,confidence_percent=64,model_image_size=12345,model_precision=fp32 person=1i 1700000000000000000',
        'prediction_output,confidence_percent=42,model_image_size=12345,model_precision=fp32 bicycle=1i 1700000000000000000',
        'prediction_output,confidence_percent=29,model_image_size=12345,model_precision=fp32 car=2i 1700000000000000000',
        'prediction_output,confidence_percent=40,model_image_size=12345,model_precision=fp32 car=1i 1700000000000000000',
    ])

  def test_callsModelWithFlagValues(self):
    YoloPredictor.predict(b'image-bytes')

    call_args = self.mock_yolo_predict.call_args
    self._assertDictContainsSubset({'imgsz': 12345, 'half': False}, call_args.kwargs)

  def test_noPredictions_skipsPredictionOutputMetrics(self):
    mock_xyxy = Mock(tolist=Mock(return_value=[]))
    mock_conf = Mock(tolist=Mock(return_value=[]))
    mock_cls = Mock(tolist=Mock(return_value=[]))
    mock_boxes = Mock(xyxy=mock_xyxy, conf=mock_conf, cls=mock_cls)
    mock_result = Mock(boxes=mock_boxes)
    mock_results = [mock_result]
    self.mock_yolo_predict = Mock(return_value=mock_results)
    self.mock_yolo = Mock(predict=self.mock_yolo_predict, names={1: 'person', 2: 'bicycle', 3: 'car'})
    YoloPredictor.set_model(self.mock_yolo)

    YoloPredictor.predict(b'image-bytes')

    self._assert_line_protocols([
        'prediction_input image_bytes=11i 1700000000000000000',
    ])
