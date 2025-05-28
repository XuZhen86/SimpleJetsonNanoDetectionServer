import json
import traceback
from email.message import Message
from enum import Enum, auto
from http.server import BaseHTTPRequestHandler

from line_protocol_cache.lineprotocolcache import LineProtocolCache

from simple_jetson_nano_detection_server.detectionrequesthandler import DetectionRequestHandler
from simple_jetson_nano_detection_server.performancetracker import PerformanceTracker


class _PerformanceCheckpoint(Enum):
  PARSE_REQUEST_BODY = auto()
  PARSE_MULTIPART_BOUNDARY = auto()
  COMPUTE_RESPONSE = auto()
  SEND_RESPONSE = auto()


class HttpRequestDispatcher(BaseHTTPRequestHandler):

  # Allows the client to query if the server is up.
  def do_HEAD(self) -> None:
    self.send_response_only(200)
    self.end_headers()

  def do_POST(self) -> None:
    if self.path != '/v1/vision/detection':
      self.send_response_only(404)
      self.end_headers()
      return

    tracker: PerformanceTracker[_PerformanceCheckpoint] = PerformanceTracker()

    try:
      with tracker(_PerformanceCheckpoint.PARSE_REQUEST_BODY):
        request_body = self._get_post_request_body()
      with tracker(_PerformanceCheckpoint.PARSE_MULTIPART_BOUNDARY):
        multipart_boundary = self._get_post_multipart_boundary()
      with tracker(_PerformanceCheckpoint.COMPUTE_RESPONSE):
        response = DetectionRequestHandler.get_response(request_body, multipart_boundary)
    except Exception as e:
      with tracker(_PerformanceCheckpoint.SEND_RESPONSE):
        self.send_response_only(400)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {
            'class': type(e).__name__,
            'message': str(e),
            'traceback': traceback.format_tb(e.__traceback__),
        }
        self.wfile.write(json.dumps(response).encode())
      LineProtocolCache.put(tracker.finalize('http_request_dispatcher', {'response_code': 400}))
      return

    with tracker(_PerformanceCheckpoint.SEND_RESPONSE):
      self.send_response_only(200)
      self.send_header('Content-Type', 'application/json')
      self.end_headers()
      self.wfile.write(response.encode())
    LineProtocolCache.put(tracker.finalize('http_request_dispatcher', {'response_code': 200}))

  def _get_post_request_body(self) -> bytes:
    content_length = int(self.headers['Content-Length'])
    assert content_length > 0, 'Expected Content-Length to be > 0'

    request_body = self.rfile.read(content_length)  # Could block forever until sufficient bytes were read.
    return request_body

  def _get_post_multipart_boundary(self) -> str:
    content_type = self.headers['Content-Type']
    assert content_type != None, 'Missing Content-Type'

    message = Message()
    message['Content-Type'] = self.headers['Content-Type']
    content_type = message.get_params()
    assert content_type != None

    mime_type = content_type[0][0]
    assert mime_type == 'multipart/form-data', f'Expected mime type to be "multipart/form-data", got "{mime_type}" instead'

    params = {p: v for p, v in content_type[1:]}
    assert 'boundary' in params, 'Missing "boundary" in Content-Type'
    return params['boundary']
