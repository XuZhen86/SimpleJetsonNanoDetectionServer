from absl.testing import parameterized

from simple_jetson_nano_detection_server.imagedataextractor import ImageDataExtractor


class TestImageDataExtractor(parameterized.TestCase):

  def test_requestBodyNoTerminatingBoundary_raises(self):
    multipart_boundary = 'boundary'
    request_body = b'\r\n'.join([
        b'',
        b'--boundary',
        b'Content-Disposition: form-data; name="image"; filename="image"',
        b'',
        b'image-data',
        b'--boundary',
        b'',
    ])

    with self.assertRaisesWithLiteralMatch(Exception, 'No terminating boundary was found'):
      ImageDataExtractor.get_first_image_data(request_body, multipart_boundary)

  def test_requestBodyNoContentDisposition_raises(self):
    multipart_boundary = 'boundary'
    request_body = b'\r\n'.join([
        b'',
        b'--boundary',
        b'Not-Content-Disposition: form-data; name="image"; filename="image"',
        b'',
        b'image-data',
        b'--boundary--',
        b'',
    ])

    with self.assertRaisesWithLiteralMatch(Exception, 'No image data was found'):
      ImageDataExtractor.get_first_image_data(request_body, multipart_boundary)

  def test_invalidContentDispositionParams_raises(self):
    multipart_boundary = 'boundary'
    request_body = b'\r\n'.join([
        b'',
        b'--boundary',
        b'Content-Disposition: not-form-data; name="image"; filename="image"',
        b'',
        b'image-data',
        b'--boundary--',
        b'',
    ])

    with self.assertRaisesWithLiteralMatch(Exception, 'No image data was found'):
      ImageDataExtractor.get_first_image_data(request_body, multipart_boundary)

  def test_validContentDisposition_returnsImageData(self):
    multipart_boundary = 'boundary'
    request_body = b'\r\n'.join([
        b'',
        b'--boundary',
        b'Content-Disposition: form-data; name="image"; filename="image"',
        b'',
        b'image-data',
        b'--boundary--',
        b'',
    ])

    self.assertEqual(ImageDataExtractor.get_first_image_data(request_body, multipart_boundary), b'image-data')

  def test_multipleHeaders_returnsFirstValidImageData(self):
    multipart_boundary = 'boundary'
    request_body = b'\r\n'.join([
        b'',
        b'--boundary',
        b'Not-Content-Disposition: form-data; name="image"; filename="image"',
        b'',
        b'image-data-1',
        b'--boundary',
        b'Content-Disposition: form-data; name="image"; filename="image"',
        b'',
        b'image-data-2',
        b'--boundary',
        b'Content-Disposition: form-data; name="image"; filename="image"',
        b'',
        b'image-data-3',
        b'--boundary--',
        b'',
    ])

    self.assertEqual(ImageDataExtractor.get_first_image_data(request_body, multipart_boundary), b'image-data-2')
