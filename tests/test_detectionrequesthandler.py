import json
from unittest.mock import Mock, patch

from absl import logging
from absl.logging.converter import absl_to_standard
from absl.testing import flagsaver, parameterized

from simple_jetson_nano_detection_server.detectionrequesthandler import _LOG_RESPONSE, DetectionRequestHandler
from simple_jetson_nano_detection_server.imagedataextractor import ImageDataExtractor
from simple_jetson_nano_detection_server.prediction import Prediction
from simple_jetson_nano_detection_server.yolopredictor import YoloPredictor

MOCK_GET_FIRST_IMAGE_DATA = Mock()
MOCK_PREDICT = Mock()


@patch.object(ImageDataExtractor, ImageDataExtractor.get_first_image_data.__name__, MOCK_GET_FIRST_IMAGE_DATA)
@patch.object(YoloPredictor, YoloPredictor.predict.__name__, MOCK_PREDICT)
class TestDetectionRequestHandler(parameterized.TestCase):

  def setUp(self):
    MOCK_GET_FIRST_IMAGE_DATA.return_value = b'image-data'
    MOCK_PREDICT.return_value = [
        Prediction(x_min=132, x_max=177, y_min=104, y_max=141, label='label-1', confidence=0.6460136771202087),
        Prediction(x_min=264, x_max=319, y_min=173, y_max=179, label='label-2', confidence=0.42441198229789734),
        Prediction(x_min=111, x_max=319, y_min=164, y_max=319, label='label-3', confidence=0.29746994376182556),
    ]

    self.saved_flags = flagsaver.as_parsed((_LOG_RESPONSE, str(False)))
    self.saved_flags.__enter__()

    return super().setUp()

  def tearDown(self) -> None:
    MOCK_GET_FIRST_IMAGE_DATA.reset_mock(return_value=True, side_effect=True)
    MOCK_PREDICT.reset_mock(return_value=True, side_effect=True)

    self.saved_flags.__exit__(None, None, None)

    return super().tearDown()

  def test_success_returnsSuccessResponse(self):
    response = DetectionRequestHandler.get_response(b'request-body', 'multipart_boundary')

    MOCK_GET_FIRST_IMAGE_DATA.assert_called_once_with(b'request-body', 'multipart_boundary')
    MOCK_PREDICT.assert_called_once_with(b'image-data')
    self.assertJsonEqual(
        response,
        json.dumps({
            'predictions': [
                {
                    'x_min': 132,
                    'x_max': 177,
                    'y_min': 104,
                    'y_max': 141,
                    'confidence': 0.6460136771202087,
                    'label': 'label-1',
                },
                {
                    'x_min': 264,
                    'x_max': 319,
                    'y_min': 173,
                    'y_max': 179,
                    'confidence': 0.42441198229789734,
                    'label': 'label-2',
                },
                {
                    'x_min': 111,
                    'x_max': 319,
                    'y_min': 164,
                    'y_max': 319,
                    'confidence': 0.29746994376182556,
                    'label': 'label-3',
                },
            ],
            'success': True,
        }))

  def test_imageDataFailure_logsAndReturnsFailureResponse(self):
    MOCK_GET_FIRST_IMAGE_DATA.side_effect = ValueError('ImageDataExtractor.get_first_image_data failed')

    with self.assertLogs(logger='absl', level=absl_to_standard(logging.ERROR)) as logs:
      response = DetectionRequestHandler.get_response(b'request-body', 'multipart_boundary')

    self.assertContainsInOrder(['Detection failed', 'ImageDataExtractor.get_first_image_data failed'], logs.output[0])
    self.assertJsonEqual(response, json.dumps({'predictions': [], 'success': False}))

  def test_predictionFailure_logsAndReturnsFailureResponse(self):
    MOCK_GET_FIRST_IMAGE_DATA.side_effect = ValueError('YoloPredictor.predict failed')

    with self.assertLogs(logger='absl', level=absl_to_standard(logging.ERROR)) as logs:
      response = DetectionRequestHandler.get_response(b'request-body', 'multipart_boundary')

    self.assertContainsInOrder(['Detection failed', 'YoloPredictor.predict failed'], logs.output[0])
    self.assertJsonEqual(response, json.dumps({'predictions': [], 'success': False}))

  @flagsaver.as_parsed((_LOG_RESPONSE, str(True)))
  def test_logResponseEnabled_logsResponse(self):
    with self.assertLogs(logger='absl', level=absl_to_standard(logging.INFO)) as logs:
      DetectionRequestHandler.get_response(b'request-body', 'multipart_boundary')

    self.assertContainsInOrder([
        "response={",
        "'predictions': [",
        "Prediction(x_min=132, x_max=177, y_min=104, y_max=141, label='label-1', confidence=0.6460136771202087), ",
        "Prediction(x_min=264, x_max=319, y_min=173, y_max=179, label='label-2', confidence=0.42441198229789734), ",
        "Prediction(x_min=111, x_max=319, y_min=164, y_max=319, label='label-3', confidence=0.29746994376182556)",
        "], ",
        "'success': True",
        "}",
    ], logs.output[0])
