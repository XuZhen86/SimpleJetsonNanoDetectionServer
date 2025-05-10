FROM ultralytics/ultralytics:8.3.127-jetson-jetpack4

WORKDIR /app
ADD . /app

RUN pip3 install --use-pep517 .
