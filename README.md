# Simple Jetson Nano Detection Server
A simple object detection server that runs on Jetson Nano.

This server:
* Uses the [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) detection model and can be configured to use other models.
* Uses TensorRT and fp16 for better performance on Jetson Nano.
* Pretends to be a [DeepStack](https://deepstack.readthedocs.io/en/latest/) server for use with [Frigate](https://docs.frigate.video/configuration/object_detectors#deepstack--codeprojectai-server-detector).
* Runs reasonably fast but does not aim to be the best performing detection server.
* Does not provide features other than simple object detection.
* Requires general knowledge on Linux, Docker, Python, and related topics.

## Approx. Inference Speed

The inference speed was measured from Frigate with these conditions:
* Jetson Nano was [running at max clock speed](https://docs.ultralytics.com/guides/nvidia-jetson/#best-practices-when-using-nvidia-jetson).
* The server was the only application using the GPU.
* No other performance-intensive applications were running on Jetson Nano.
* Frigate was running on a different x64 machine.
* Frigate was interacting with the server via wired LAN.

| Model     | Input Size | Precision | Inference Speed per Image |
|-----------|------------|-----------|---------------------------|
| `yolo11n` | 320x320    | fp16      | 20ms                      |
| `yolo11s` | 320x320    | fp16      | 50ms                      |
| `yolo11m` | 320x320    | fp16      | 100ms                     |

`yolo11s` was selected as the default model because it offers an acceptable balance between accuracy and speed.

## Running the Server

There are a few steps needed to run the server.
Depending on the exact configuration on your Jetson Nano, you may skip some of the steps.

Make sure to read and understand the instructions before executing the commands.

### Update the Docker Daemon Config

For the container to use the Jetson Nano GPU, we must tell Docker to use the Nvidia container runtime when using the `docker run` command.

However, the `docker-compose` command that comes with Jetpack 4.6.3 reads from the `docker-compose.yml` file but does support specifying the container runtime in the YAML file.

To mitigate this limitation, we can tell Docker to always use Nvidia container runtime by adding the `"default-runtime": "nvidia"` to the `/etc/docker/daemon.json` file.

The `/etc/docker/daemon.json` file should look like this:
```
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

After editing, restart Docker by running `sudo systemctl restart docker`.

### Build the Docker Image

Run `make docker-image` and wait for it to finish.

As part of the building process, Docker will download a 5.5GB [Ultralytics image](https://hub.docker.com/r/ultralytics/ultralytics).
Make sure you have sufficient free space on the SD card before building.

After building has finished, run `docker image ls` and make sure we have two new images:
```
REPOSITORY                            TAG                       IMAGE ID       CREATED         SIZE
simple-jetson-nano-detection-server   latest                    6fe14627290c   5 minutes ago   5.59GB
ultralytics/ultralytics               8.3.127-jetson-jetpack4   29afc116eedc   4 weeks ago     5.5GB
```

### Create the Docker Volume

A persistent storage location is needed to store the YOLO11 model files.
In this case, a Docker volume is specified in the `docker-compose.yml` file and it's used to store the files.

Run `docker-compose create` to create the volume. It will also create two containers, leave them alone for now and we will get back to them later.

Run `docker volume ls` and make sure we have a new volume:
```
DRIVER    VOLUME NAME
local     simplejetsonnanodetectionserver_prod-data
```

### Export the Model

The project comes with a Bash script to export the model files.
The Bash script exports the `yolo11n`, `yolo11s`, and `yolo11m` models.
Make sure you have at least 1GB free space on the SD card before exporting.

The script can take up to 30 minutes to finish.

Run the following command to export the model:
```
$ docker run \
  --interactive \
  --tty \
  --rm \
  --ipc=host \
  -v simplejetsonnanodetectionserver_prod-data:/app/data \
  simple-jetson-nano-detection-server:latest \
  /app/export-tensorrt-engines.sh
```

The Bash script does not attempt to export `yolo11l` and `yolo11x` models, because they appear to require more memory than Jetson Nano offers.

### Create the Config File

The server can be configured via a set of flags, and it loads a flag file on startup.
For now, we can create an empty flag file and keep everything default.

Run the following two commands to create an empty flag file:
```
$ docker run \
  --interactive \
  --tty \
  --rm \
  -v simplejetsonnanodetectionserver_prod-data:/app/data \
  simple-jetson-nano-detection-server:latest \
  mkdir --parents /app/data/flags
$ docker run \
  --interactive \
  --tty \
  --rm \
  -v simplejetsonnanodetectionserver_prod-data:/app/data \
  simple-jetson-nano-detection-server:latest \
  touch /app/data/flags/detection-server.txt
```

### Start the Server

Now we should be ready to run the server.
Run `docker-compose up prod-detection-server` and monitor the logs.

The logs should look like:
```
Attaching to simplejetsonnanodetectionserver_prod-detection-server_1
prod-detection-server_1  | Creating new Ultralytics Settings v0.0.6 file ✅
prod-detection-server_1  | View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
prod-detection-server_1  | Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
prod-detection-server_1  | I0605 08:11:15.135380 548040200208 lineprotocolcache.py:54] Starting LineProtocolCache thread.
prod-detection-server_1  | I0605 08:11:15.346673 548040200208 lineprotocolcache.py:57] Thread @13 has started.
prod-detection-server_1  | Loading data/yolo11/models/tensorrt/yolo11s-320-fp16.engine for TensorRT inference...
prod-detection-server_1  | [06/05/2025-08:11:19] [TRT] [I] [MemUsageChange] Init CUDA: CPU +230, GPU +0, now: CPU 300, GPU 2554 (MiB)
prod-detection-server_1  | [06/05/2025-08:11:19] [TRT] [I] Loaded engine size: 21 MiB
prod-detection-server_1  | [06/05/2025-08:11:22] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +158, GPU +255, now: CPU 488, GPU 2857 (MiB)
prod-detection-server_1  | [06/05/2025-08:11:25] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +240, GPU +360, now: CPU 728, GPU 3217 (MiB)
prod-detection-server_1  | [06/05/2025-08:11:25] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +20, now: CPU 0, GPU 20 (MiB)
prod-detection-server_1  | [06/05/2025-08:11:25] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +0, now: CPU 707, GPU 3196 (MiB)
prod-detection-server_1  | [06/05/2025-08:11:25] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +0, now: CPU 707, GPU 3196 (MiB)
prod-detection-server_1  | [06/05/2025-08:11:25] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +21, now: CPU 0, GPU 41 (MiB)
```

If there are no error messages, then we are in good shape.
Press Ctrl+C to stop the server, then run `docker-compose up -d prod-detection-server` to run the server in the background.

Run `docker-compose logs --follow --tail=10` to monitor the server logs.

### Configure Frigate to Use the Server

Once the server is running, we can configure Frigate to talk to the server.

Example Frigate config to be added:
```
detectors:
  deepstack:
    api_url: http://192.168.1.3:32168/v1/vision/detection
    type: deepstack
    api_timeout: 1.0
```

Restart Frigate, navigate to "System metrics/General" tab, and the Detector section should show a "deepstack" detector being used.


## Server Configuration

The server can be configured with command line flags.
By default, the server reads command line flags from `/app/data/flags/detection-server.txt`.

Example content for `detection-server.txt`:
```
--engine_path=data/yolo11/models/tensorrt/yolo11s-320-fp16.engine
--image_size=320
--half_precision=true
--server_ip=0.0.0.0
--server_port=32168
--log_response=false
--generate_metrics=false
```

Available command line flags:
```
simple_jetson_nano_detection_server.detectionrequesthandler:
  --[no]log_response: If true, log the detection response
    (default: 'false')

simple_jetson_nano_detection_server.imagedataextractor:
  --max_image_data_bytes: Maximum image size in bytes that is allowed. The value is inclusive
    (default: '65536')
    (a non-negative integer)

simple_jetson_nano_detection_server.main:
  --engine_path: Path to the exported TensorRT engine file
    (default: 'data/yolo11/models/tensorrt/yolo11s-320-fp16.engine')
  --[no]generate_metrics: Generate InfluxDB data points when processing the requests
    (default: 'false')
  --server_ip: The IP address to bind the HTTP server to
    (default: '0.0.0.0')
  --server_port: The port to bind the HTTP server to
    (default: '32168')
    (an integer)

simple_jetson_nano_detection_server.yolopredictor:
  --[no]half_precision: Set to true if the TensorRT engine file was exported with FP16. Jetson Nano runs faster with 16-bit floating point numbers. Passed to the "half" argument
    (default: 'true')
  --image_size: The image size used when exporting the TensorRT engine file. Passed to the "imgsz" argument
    (default: '320')
    (an integer)
```

## Server Metrics

When setting `--generate_metrics=true`, the server generates metrics that can be imported into InfluxDB.
The server uses the [LineProtocolCache](https://github.com/XuZhen86/LineProtocolCache) library to persist the data points in the Docker volume, then a second container running the LineProtocolCache uploader to upload the data points to an InfluxDB server.

The LineProtocolCache uploader can be configured with `/app/data/flags/metrics-uploader.txt`.
Start the uploader container by running `docker-compose up prod-metrics-uploader`.

Example content for `metrics-uploader.txt`:
```
// The InfluxDB server address and port.
--urls=http://192.168.1.2:8086
// The InfluxDB bucket name.
--buckets=My Bucket
// The token with write permission to the bucket.
--tokens=XsxuB729WdV6sq9HZqg25YzOg15DXB91xI309BK0V5BuZmRHdk66UCeqD9C1G05Gxim97G3i83S6qqA19vP2p==
// The name of the organization that hosts the bucket.
--orgs=My Org
// Duration in seconds between printing a data point for debugging.
--sample_interval_s=10
--verbosity=0
```

## Use a Different Model and Considerations

You can specify the path to the exported TensorRT engine file using the flag `--engine_path`.
Additionally, you may need to specify `--image_size` and `--half_precision` to match the parameter of the TensorRT engine.

Jetson Nano offers 471.6 GFLOPS of fp16 performance, but drops to 235.8 GFLOPS for fp32.
The model should be exported with fp16 for better performance.
If running with a fp32 model, set `--half_precision=false`.
Jetson Nano does not support int8.

Frigate appears to always send the image of size 320x320, while the pre-trained Ultralytics models support 640x640.
The model should be exported with input size of 320x320 for better performance.
If the input image dimension is different from what the model was exported with, the Ultralytics Python library will automatically resize the image and it should not fail the detection.

The pre-trained model can be [exported as different formats](https://docs.ultralytics.com/modes/export/#export-formats), but TensorRT [runs the fastest](https://docs.ultralytics.com/guides/nvidia-jetson/#use-tensorrt-on-nvidia-jetson) on a Jetson Nano.
For simplicity, the server only supports running with a TensorRT engine file.

## HTTP Endpoints

The server exposes two HTTP endpoints:
* `POST /v1/vision/detection`: For object detection.
It mimics the same endpoint used in [DeepStack](https://deepstack.readthedocs.io/en/latest/api-reference/index.html#object-detection).
* `HEAD /`: For the client to check if the server is running.
The server always responds HTTP 200 with an empty body.

Since `/v1/vision/detection` is the only heavy-lifting endpoint, we will be referring to it as "the endpoint" for the rest of the doc.

### Detection Request

The endpoint expects a [Multipart form submission](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Methods/POST#multipart_form_submission) that contains the JPG image bytes.

The server processes the request in the following fashion:
1. Iterates over each part in the request body,
1. Looks for the first part that matches the header `Content-Disposition: form-data; name="image"; filename="image"`
1. Runs the object detection on that part,
1. Produces a JSON response,
1. Early-returns with the response.

The server ignores all subsequent request body parts, even if they match the header.

Example request header:
```
Content-Type: multipart/form-data; boundary="boundary30729552400834427008111221218144"
```

Example request body, the extra new lines in this example are required:
```

--boundary30729552400834427008111221218144
Content-Disposition: form-data; name="image"; filename="image"

The JPG image data goes here.
--boundary30729552400834427008111221218144--

```

### Success Response

If the HTTP body parsing and object detection were successful, the endpoint returns an HTTP 200 response with a JSON response body.
The JSON response contains the box coordinates, labels, confidences for each detected object.

Example success response:
```
{
  "predictions": [
    { "x_min": 132, "x_max": 177, "y_min": 104, "y_max": 141, "label": "car", "confidence": 0.6460136771202087 },
    { "x_min": 264, "x_max": 319, "y_min": 173, "y_max": 179, "label": "person", "confidence": 0.42441198229789734 },
    { "x_min": 111, "x_max": 319, "y_min": 164, "y_max": 319, "label": "car", "confidence": 0.29746994376182556 }
  ],
  "success": true
}
```

### Failure Response for HTTP Parsing Error

If the server is unable to parse the HTTP request, the endpoint returns an HTTP 400 response with a JSON response body.
The JSON response describes the culprit Python exception.

Example failure response:
```
{
  "class": "AssertionError",
  "message": "Expected mime type to be \"multipart/form-data\", got \"invalid/mime-type\" instead",
  "traceback": [
    "  File \"/app/simple_jetson_nano_detection_server/httprequesdispatcher.py\", line 39, in do_POST\n    multipart_boundary = self._get_post_multipart_boundary()\n",
    "  File \"/app/simple_jetson_nano_detection_server/httprequesdispatcher.py\", line 80, in _get_post_multipart_boundary\n    assert mime_type == 'multipart/form-data', f'Expected mime type to be \"multipart/form-data\", got \"{mime_type}\" instead'\n"
  ]
}
```

### Failure Response for Detection Error

If the server was able to parse the HTTP request, but the detection failed, the endpoint returns an HTTP 200 response with a JSON response body.

Unlike during an HTTP parsing error, in which the server returns the error to the client, the server logs the culprit Python exception instead.

Failure response (the server returns this exact response):
```
{
  "predictions": [],
  "success": false
}
```

## Related Topics

Motivations for this project:
* Frigate 0.16 is removing support for Jetpack 4, which is required for Jetson Nano.
* The new Jetson Orin Nano Super is unobtainium at the time of writing.
* [DeepStack](https://github.com/johnolafenwa/DeepStack) is not being maintained anymore, does not appear to be using TensorRT, and is using YOLOv5.
* [CodeProject.AI Server](https://codeproject.github.io/codeproject.ai/index.html) does not support CUDA 10, which is required for Jetson Nano.
* The 4GB memory on Jetson Nano is too tight for running Frigate but sufficient for running a small detection server.
