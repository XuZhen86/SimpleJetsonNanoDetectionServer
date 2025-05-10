import setuptools

setuptools.setup(
    name='SimpleJetsonNanoDetectionServer',
    version='0.1',
    author='XuZhen86',
    url='https://github.com/XuZhen86/SimpleJetsonNanoDetectionServer',
    packages=setuptools.find_packages(),
    python_requires='>=3.8.0',
    install_requires=[
        'absl-py==2.1.0',
        'numpy==1.23.5',  # Required when running the model.
        'onnxslim>=0.1.46',  # Required when exporting the model.
    ],
    entry_points={
        'console_scripts': [
            'simple-jetson-nano-detection-server = simple_jetson_nano_detection_server.main:app_run_main',
        ],
    },
)
