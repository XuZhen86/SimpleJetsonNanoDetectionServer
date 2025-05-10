import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import NamedTemporaryFile
from typing import List, Tuple

from absl import app, flags, logging
from ultralytics import YOLO

from simple_jetson_nano_detection_server.prediction import Prediction, PredictionJsonEncoder

ENGINE_PATH = flags.DEFINE_string(
    name='engine_path',
    default='data/yolo11/models/tensorrt/yolo11s-320-fp16.engine',
    help='Path to the exported TensorRT engine file',
)

IMAGE_SIZE = flags.DEFINE_integer(
    name='image_size',
    default=320,
    help='The image size used when exporting the TensorRT engine file. '
    'Passed to the "imgsz" argument',
)

HALF_PRECISION = flags.DEFINE_bool(
    name='half_precision',
    default=True,
    help='Set to true if the TensorRT engine file was exported with FP16. '
    'Jetson Nano runs faster with 16-bit floating point numbers. '
    'Passed to the "half" argument',
)

SERVER_IP = flags.DEFINE_string(
    name='server_ip',
    default='0.0.0.0',
    help='The IP address to bind the HTTP server to',
)

SERVER_PORT = flags.DEFINE_integer(
    name='server_port',
    default=32168,
    help='The port to bind the HTTP server to',
)

LOG_RESPONSE = flags.DEFINE_bool(
    name='log_response',
    default=False,
    help='If true, log the HTTP response body',
)


class DetectionRequestHandler(BaseHTTPRequestHandler):
  _MULTIPART_BODY_HEADER = b'\r\nContent-Disposition: form-data; name="image"; filename="image"\r\n\r\n'
  _MODEL: YOLO

  def do_POST(self) -> None:
    if self.path != '/v1/vision/detection':
      self.send_response(404)
      return

    predictions: List[Prediction] = []
    success = True

    try:
      predictions = self._get_predictions()
    except Exception:
      logging.exception('Error when getting predictions, sending failure response')
      success = False

    response = json.dumps({'predictions': predictions, 'success': success}, cls=PredictionJsonEncoder)
    if LOG_RESPONSE.value:
      logging.info(f'{response=}')

    self.send_response(200)
    self.send_header('Content-type', 'application/json')
    self.end_headers()
    self.wfile.write(response.encode())

  def _get_predictions(self) -> List[Prediction]:
    image_data = self._get_image_data()
    with NamedTemporaryFile(dir='/dev/shm', suffix='.jpg') as image_file:
      image_file.write(image_data)
      image_file.flush()
      results = self._MODEL.predict(image_file.name, imgsz=IMAGE_SIZE.value, half=HALF_PRECISION.value, save=False)

    assert results[0].boxes != None
    zipped: zip[Tuple[List[float], float, float]] = zip(
        # [[132.5, 104.5, 177.5, 141.875], [264.25, 73.40625, 319.75, 112.09375], [111.5, 164.5625, 319.5, 319.4375]]
        results[0].boxes.xyxy.tolist(),
        # [0.6460136771202087, 0.42441198229789734, 0.29746994376182556]
        results[0].boxes.conf.tolist(),
        # [2.0, 2.0, 2.0]
        results[0].boxes.cls.tolist(),
    )

    predictions: List[Prediction] = []
    for xyxy_coordinate, confidence, class_id in zipped:
      predictions.append(
          Prediction(x_min=int(xyxy_coordinate[0]),
                     y_min=int(xyxy_coordinate[1]),
                     x_max=int(xyxy_coordinate[2]),
                     y_max=int(xyxy_coordinate[3]),
                     confidence=float(confidence),
                     label=str(self._MODEL.names.get(int(class_id)))))

    return predictions

  def _get_image_data(self) -> bytes:
    post_content = self._get_post_content()
    boundary = self._get_multipart_boundary()

    parts = post_content.split(b'--' + boundary)
    assert parts[-1] == b'--\r\n'

    for part in parts:
      if not part.startswith(self._MULTIPART_BODY_HEADER):
        continue
      return part[len(self._MULTIPART_BODY_HEADER):-len(b'\r\n')]

    raise ValueError('No multipart body was found to contain image data.')

  def _get_post_content(self) -> bytes:
    content_length = int(self.headers['Content-Length'])
    post_content = self.rfile.read(content_length)
    return post_content

  def _get_multipart_boundary(self) -> bytes:
    # 'multipart/form-data; boundary=241a860e9a94d2780e8e67095c27a662'
    content_type = str(self.headers['Content-Type'])

    # 'multipart/form-data; '
    # '241a860e9a94d2780e8e67095c27a662'
    multipart_form_data, boundary = content_type.split('boundary=')
    assert multipart_form_data.startswith('multipart/form-data;')

    return boundary.encode()


def main(args: List[str]) -> None:
  # Load the engine file.
  model = YOLO(ENGINE_PATH.value, task='detect')
  # Do one prediction to load the engine into GPU.
  model.predict("images/bus.jpg", imgsz=IMAGE_SIZE.value, half=HALF_PRECISION.value, save=False)

  DetectionRequestHandler._MODEL = model
  http_server = HTTPServer((SERVER_IP.value, SERVER_PORT.value), DetectionRequestHandler)
  http_server.serve_forever()


def app_run_main() -> None:
  app.run(main)
