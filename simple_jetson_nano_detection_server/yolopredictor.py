from enum import Enum, auto
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import ultralytics
from absl import flags
from line_protocol_cache.lineprotocolcache import LineProtocolCache

from simple_jetson_nano_detection_server.cocolabel import CocoLabel
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


class YoloPredictor:

  _model: Optional[ultralytics.YOLO] = None

  @classmethod
  def set_model(cls, model: ultralytics.YOLO) -> None:
    cls._model = model

  @classmethod
  def predict(cls, image_data: bytes) -> List[Prediction]:
    assert cls._model is not None, 'A model must be set before prediction'

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
          Prediction.build(
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

    tracker: EventMetricsTracker[CocoLabel] = EventMetricsTracker()
    for prediction in predictions:
      tracker.increment(prediction.label, 1, {'confidence_percent': int(prediction.confidence * 100)})

    LineProtocolCache.put(
        tracker.finalize('prediction_output', {
            'model_image_size': _IMAGE_SIZE.value,
            'model_precision': 'fp16' if _HALF_PRECISION.value else 'fp32',
        }))
