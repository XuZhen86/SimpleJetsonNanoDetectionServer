from enum import Enum, auto
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import ultralytics
from absl import flags
from line_protocol_cache.lineprotocolcache import LineProtocolCache

from simple_jetson_nano_detection_server.eventmetricstracker import EventMetricsTracker
from simple_jetson_nano_detection_server.prediction import Prediction

_IMAGE_SIZE = flags.DEFINE_integer(
    name='image_size',
    default=320,
    help='The image size used when exporting the TensorRT engine file. '
    'Passed to the "imgsz" argument',
)

_HALF_PRECISION = flags.DEFINE_bool(
    name='half_precision',
    default=True,
    help='Set to true if the TensorRT engine file was exported with FP16. '
    'Jetson Nano runs faster with 16-bit floating point numbers. '
    'Passed to the "half" argument',
)


class _EventMetricsFields(Enum):
  IMAGE_BYTES = auto()


class _CocoCategories(Enum):
  PERSON = 'person'
  BICYCLE = 'bicycle'
  CAR = 'car'
  MOTORCYCLE = 'motorcycle'
  AIRPLANE = 'airplane'
  BUS = 'bus'
  TRAIN = 'train'
  TRUCK = 'truck'
  BOAT = 'boat'
  TRAFFIC_LIGHT = 'traffic light'
  FIRE_HYDRANT = 'fire hydrant'
  STOP_SIGN = 'stop sign'
  PARKING_METER = 'parking meter'
  BENCH = 'bench'
  BIRD = 'bird'
  CAT = 'cat'
  DOG = 'dog'
  HORSE = 'horse'
  SHEEP = 'sheep'
  COW = 'cow'
  ELEPHANT = 'elephant'
  BEAR = 'bear'
  ZEBRA = 'zebra'
  GIRAFFE = 'giraffe'
  BACKPACK = 'backpack'
  UMBRELLA = 'umbrella'
  HANDBAG = 'handbag'
  TIE = 'tie'
  SUITCASE = 'suitcase'
  FRISBEE = 'frisbee'
  SKIS = 'skis'
  SNOWBOARD = 'snowboard'
  SPORTS_BALL = 'sports ball'
  KITE = 'kite'
  BASEBALL_BAT = 'baseball bat'
  BASEBALL_GLOVE = 'baseball glove'
  SKATEBOARD = 'skateboard'
  SURFBOARD = 'surfboard'
  TENNIS_RACKET = 'tennis racket'
  BOTTLE = 'bottle'
  WINE_GLASS = 'wine glass'
  CUP = 'cup'
  FORK = 'fork'
  KNIFE = 'knife'
  SPOON = 'spoon'
  BOWL = 'bowl'
  BANANA = 'banana'
  APPLE = 'apple'
  SANDWICH = 'sandwich'
  ORANGE = 'orange'
  BROCCOLI = 'broccoli'
  CARROT = 'carrot'
  HOT_DOG = 'hot dog'
  PIZZA = 'pizza'
  DONUT = 'donut'
  CAKE = 'cake'
  CHAIR = 'chair'
  COUCH = 'couch'
  POTTED_PLANT = 'potted plant'
  BED = 'bed'
  DINING_TABLE = 'dining table'
  TOILET = 'toilet'
  TV = 'tv'
  LAPTOP = 'laptop'
  MOUSE = 'mouse'
  REMOTE = 'remote'
  KEYBOARD = 'keyboard'
  CELL_PHONE = 'cell phone'
  MICROWAVE = 'microwave'
  OVEN = 'oven'
  TOASTER = 'toaster'
  SINK = 'sink'
  REFRIGERATOR = 'refrigerator'
  BOOK = 'book'
  CLOCK = 'clock'
  VASE = 'vase'
  SCISSORS = 'scissors'
  TEDDY_BEAR = 'teddy bear'
  HAIR_DRIER = 'hair drier'
  TOOTHBRUSH = 'toothbrush'


class YoloPredictor:

  _model: Optional[ultralytics.YOLO] = None

  @classmethod
  def set_model(cls, model: ultralytics.YOLO) -> None:
    cls._model = model

  @classmethod
  def predict(cls, image_data: bytes) -> List[Prediction]:
    assert cls._model is not None, 'A model must be set before prediciton'

    cls._record_image_size(image_data)
    with NamedTemporaryFile(dir='/dev/shm', suffix='.jpg') as image_file:
      image_file.write(image_data)
      image_file.flush()
      results = cls._model.predict(image_file.name,
                                   imgsz=_IMAGE_SIZE.value,
                                   half=_HALF_PRECISION.value,
                                   save=False,
                                   verbose=False)

    assert len(results) == 1, f'There must be exactly 1 result, got {len(results)} instead'
    result = results[0]

    assert result.boxes != None, 'Boxes cannot be None'
    zipped: zip[Tuple[List[float], float, float]] = zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
    )

    predictions: List[Prediction] = []
    for xyxy_coordinate, confidence, class_id in zipped:
      predictions.append(
          Prediction(
              x_min=int(xyxy_coordinate[0]),
              y_min=int(xyxy_coordinate[1]),
              x_max=int(xyxy_coordinate[2]),
              y_max=int(xyxy_coordinate[3]),
              confidence=float(confidence),
              label=str(cls._model.names.get(int(class_id))),
          ))

    cls._record_coco_categories(predictions)
    return predictions

  @classmethod
  def _record_image_size(cls, image_data: bytes) -> None:
    tracker: EventMetricsTracker[_EventMetricsFields] = EventMetricsTracker()
    tracker.record(_EventMetricsFields.IMAGE_BYTES, len(image_data))
    LineProtocolCache.put(tracker.finalize('prediction_input'))

  @classmethod
  def _record_coco_categories(cls, predictions: List[Prediction]) -> None:
    if len(predictions) == 0:
      return

    tracker: EventMetricsTracker[_CocoCategories] = EventMetricsTracker()
    for prediction in predictions:
      tracker.increment(_CocoCategories(prediction.label), 1, {'confidence_percent': int(prediction.confidence * 100)})

    LineProtocolCache.put(
        tracker.finalize('prediction_output', {
            'model_image_size': _IMAGE_SIZE.value,
            'model_precision': 'fp16' if _HALF_PRECISION.value else 'fp32',
        }))
