from unittest.mock import Mock

from absl.testing import flagsaver, parameterized

from simple_jetson_nano_detection_server.prediction import Prediction
from simple_jetson_nano_detection_server.yolopredictor import _HALF_PRECISION, _IMAGE_SIZE, YoloPredictor


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
    ]))
    mock_conf = Mock(tolist=Mock(return_value=[
        0.6460136771202087,
        0.42441198229789734,
        0.29746994376182556,
    ]))
    mock_cls = Mock(tolist=Mock(return_value=[
        1.0,
        2.0,
        3.0,
    ]))
    mock_boxes = Mock(xyxy=mock_xyxy, conf=mock_conf, cls=mock_cls)
    mock_result = Mock(boxes=mock_boxes)
    mock_results = [mock_result]

    self.mock_yolo_predict = Mock(return_value=mock_results)
    self.mock_yolo = Mock(predict=self.mock_yolo_predict, names={1: 'label-1', 2: 'label-2', 3: 'label-3'})

    YoloPredictor.set_model(self.mock_yolo)

    return super().setUp()

  def tearDown(self) -> None:
    self.saved_flags.__exit__(None, None, None)
    return super().tearDown()

  def test_noModel_raises(self):
    YoloPredictor._model = None

    with self.assertRaisesWithLiteralMatch(Exception, 'A model must be set before prediciton'):
      YoloPredictor.predict(b'')

  def test_noResults_raises(self):
    self.mock_yolo_predict.return_value = []

    with self.assertRaisesWithLiteralMatch(Exception, 'There must be exactly 1 result, got 0 instead'):
      YoloPredictor.predict(b'')

  def test_moreThan1Results_raises(self):
    self.mock_yolo_predict.return_value = [Mock(), Mock()]

    with self.assertRaisesWithLiteralMatch(Exception, 'There must be exactly 1 result, got 2 instead'):
      YoloPredictor.predict(b'')

  def test_boxesIsNone_raises(self):
    self.mock_yolo_predict.return_value = [Mock(boxes=None)]

    with self.assertRaisesWithLiteralMatch(Exception, 'Boxes cannot be None'):
      YoloPredictor.predict(b'')

  def test_convertsToPredictions(self):
    predictions = YoloPredictor.predict(b'')

    self.assertEqual(predictions, [
        Prediction(x_min=132, x_max=177, y_min=104, y_max=141, label='label-1', confidence=0.6460136771202087),
        Prediction(x_min=264, x_max=319, y_min=173, y_max=179, label='label-2', confidence=0.42441198229789734),
        Prediction(x_min=111, x_max=319, y_min=164, y_max=319, label='label-3', confidence=0.29746994376182556),
    ])

  def test_callsModelWithFlagValues(self):
    YoloPredictor.predict(b'')

    call_args = self.mock_yolo_predict.call_args
    self.assertContainsSubset({'imgsz': 12345, 'half': False}, call_args.kwargs)
