from email.message import Message

from absl import flags

_MAX_IMAGE_DATA_BYTES = flags.DEFINE_integer(
    name='max_image_data_bytes',
    default=64 * 1024,  # The observed image size goes up to 32KiB.
    lower_bound=0,
    help='Maximum image size in bytes that is allowed. The value is inclusive',
)


class ImageDataExtractor:

  @classmethod
  def get_first_image_data(cls, request_body: bytes, multipart_boundary: str) -> bytes:
    boundary = b'--' + multipart_boundary.encode()

    parts = request_body.split(boundary)
    assert parts[-1] == b'--\r\n', 'No terminating boundary was found'

    for part in parts:
      start_index = part.find(b'\r\nContent-Disposition:')
      if start_index == -1:
        continue
      start_index += len(b'\r\nContent-Disposition:')

      end_index = part.find(b'\r\n', start_index)
      if end_index == -1:
        continue

      header = part[start_index:end_index]
      message = Message()
      message['Content-Type'] = header.decode()  # Using Content-Type to trick Message into parsing the header.
      content_disposition = message.get_params()

      if content_disposition != [('form-data', ''), ('name', 'image'), ('filename', 'image')]:
        continue

      image_data = part[end_index + len(b'\r\n\r\n'):-len(b'\r\n')]
      assert len(image_data) <= _MAX_IMAGE_DATA_BYTES.value, (f'Image size of {len(image_data)} bytes is too big, '
                                                              f'must be <= {_MAX_IMAGE_DATA_BYTES.value} bytes')

      return image_data

    raise ValueError('No image data was found')
