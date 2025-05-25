import time
from http.server import HTTPServer
from multiprocessing import Manager, Process
from queue import Queue
from typing import Any, Tuple
from unittest.mock import Mock, patch

import requests
from absl.testing import parameterized

from simple_jetson_nano_detection_server.detectionrequesthandler import DetectionRequestHandler
from simple_jetson_nano_detection_server.httprequesdispatcher import HttpRequestDispatcher


class TestHttpRequestDispatcher(parameterized.TestCase):
  SERVER_IP = '127.0.0.1'
  SERVER_PORT = 42069

  def setUp(self):
    self.manager = Manager()
    self.call_args = self.manager.Queue()

    self.server_process = Process(target=self._run_server, args=(self.call_args,))
    self.server_process.start()

    for _ in range(100):
      try:
        requests.head(f'http://{self.SERVER_IP}:{self.SERVER_PORT}').raise_for_status()
        return super().setUp()
      except Exception:
        time.sleep(0.01)

    raise TimeoutError('HTTP server did not become ready')

  @classmethod
  def _run_server(cls, call_args: 'Queue[Any]') -> None:

    def put_call_args(*args: Tuple[Any, ...]) -> str:
      call_args.put(args)
      return ''

    mock = Mock(side_effect=put_call_args)

    with patch.object(DetectionRequestHandler, DetectionRequestHandler.get_response.__name__, mock):
      server = HTTPServer((cls.SERVER_IP, cls.SERVER_PORT), HttpRequestDispatcher)
      server.serve_forever()

  def tearDown(self):
    self.server_process.terminate()
    self.server_process.join(timeout=5)
    assert self.server_process.exitcode is not None, 'Failed to terminate server process'
    return super().tearDown()

  def test_invalidPath_returns404(self):
    r = requests.post(f'http://{self.SERVER_IP}:{self.SERVER_PORT}/invalid-path')

    self.assertEqual(r.status_code, 404)

  def test_emptyBody_returns400(self):
    r = requests.post(
        f'http://{self.SERVER_IP}:{self.SERVER_PORT}/v1/vision/detection',
        headers={'Content-Type': 'multipart/form-data; boundary=241a860e9a94d2780e8e67095c27a662'},
        data=b'',
    )

    self.assertEqual(r.status_code, 400)
    self.assertContainsSubset({'message': 'Expected Content-Length to be > 0'}, r.json())

  def test_noContentTypeHeader_returns400(self):
    r = requests.post(
        f'http://{self.SERVER_IP}:{self.SERVER_PORT}/v1/vision/detection',
        headers={'not-Content-Type': 'multipart/form-data; boundary=241a860e9a94d2780e8e67095c27a662'},
        data=b'12345',
    )

    self.assertEqual(r.status_code, 400)
    self.assertContainsSubset({'message': 'Missing Content-Type'}, r.json())

  def test_invalidMimeType_returns400(self):
    r = requests.post(
        f'http://{self.SERVER_IP}:{self.SERVER_PORT}/v1/vision/detection',
        headers={'Content-Type': 'invalid/mime-type; boundary=241a860e9a94d2780e8e67095c27a662'},
        data=b'12345',
    )

    self.assertEqual(r.status_code, 400)
    self.assertContainsSubset(
        {'message': 'Expected mime type to be "multipart/form-data", got "invalid/mime-type" instead'}, r.json())

  def test_noBoundary_returns400(self):
    r = requests.post(
        f'http://{self.SERVER_IP}:{self.SERVER_PORT}/v1/vision/detection',
        headers={'Content-Type': 'multipart/form-data; not-boundary=241a860e9a94d2780e8e67095c27a662'},
        data=b'12345',
    )

    self.assertEqual(r.status_code, 400)
    self.assertContainsSubset({'message': 'Missing "boundary" in Content-Type'}, r.json())

  def test_validRequest_callsHandler(self):
    r = requests.post(
        f'http://{self.SERVER_IP}:{self.SERVER_PORT}/v1/vision/detection',
        headers={'Content-Type': 'multipart/form-data; boundary=241a860e9a94d2780e8e67095c27a662'},
        data=b'12345',
    )

    self.assertEqual(r.status_code, 200)
    self.assertEqual(self.call_args.get(timeout=5), (b'12345', '241a860e9a94d2780e8e67095c27a662'))
