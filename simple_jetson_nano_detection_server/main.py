from http.server import HTTPServer
from typing import List

from absl import app, flags
from ultralytics import YOLO

from simple_jetson_nano_detection_server.httprequesdispatcher import HttpRequestDispatcher
from simple_jetson_nano_detection_server.yolopredictor import YoloPredictor

ENGINE_PATH = flags.DEFINE_string(
    name='engine_path',
    default='data/yolo11/models/tensorrt/yolo11s-320-fp16.engine',
    help='Path to the exported TensorRT engine file',
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


def main(args: List[str]) -> None:
  # Load the engine file.
  model = YOLO(ENGINE_PATH.value, task='detect')
  YoloPredictor.set_model(model)

  # Do one prediction to load the engine into GPU.
  with open("images/bus.jpg", 'rb') as fp:
    YoloPredictor.predict(fp.read())

  http_server = HTTPServer((SERVER_IP.value, SERVER_PORT.value), HttpRequestDispatcher)
  http_server.serve_forever()


def app_run_main() -> None:
  app.run(main)
