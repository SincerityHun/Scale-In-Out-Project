# Dockerfile
FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1


# Get example python scripts
# pytorch_mnist_elastic with simple CNN Model
RUN touch /examples/elastic/sincerityhun_pytorch_mnist_elastic.py
COPY pytorch_mnist_elastic.py /examples/elastic/sincerityhun_pytorch_mnist_elastic.py

# tensorflow2_mnist_elastic  with simple CNN Model
RUN touch /examples/elastic/sincerityhun_tensorflow2_mnist_elastic.py
COPY tensorflow2_mnist_elastic.py /examples/elastic/sincerityhun_tensorflow2_mnist_elastic.py

# tensorflow2_mnist_elastic  with ResNet50 Model

# tensorflow2_mnist_elastic  with VGG16 Model

# tensorflow2_mnist_elastic wit