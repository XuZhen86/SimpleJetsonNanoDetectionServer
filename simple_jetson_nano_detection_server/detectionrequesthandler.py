import json

from absl import flags, logging

from simple_jetson_nano_detection_server.imagedataextractor import ImageDataExtractor
from simple_jetson_nano_detection_server.prediction import PredictionJsonEncoder
from simple_jetson_nano_detection_server.yolopredictor import YoloPredictor

_LOG_RESPONSE = flags.DEFINE_bool(
    name='log_response',
    default=False,
    help='If true, log the detection response',
)


class DetectionRequestHandler:

  @classmethod
  def get_response(cls, request_body: bytes, multipart_boundary: str) -> str:
    try:
      image_data = ImageDataExtractor.get_first_image_data(request_body, multipart_boundary)
      predictions = YoloPredictor.predict(image_data)
      response = {'predictions': predictions, 'success': True}
    except Exception:
      logging.exception('Detection failed')
      response = {'predictions': [], 'success': False}

    if _LOG_RESPONSE.value:
      logging.info(f'{response=}')

    return json.dumps(response, cls=PredictionJsonEncoder)
