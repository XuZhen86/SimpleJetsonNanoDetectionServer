class DetectionRequestHandler:

  @classmethod
  def get_response(cls, request_body: bytes, multipart_boundary: str) -> str:
    raise NotImplementedError()
