# NVIDIA-DOCKER 확인용 

# docker build -t {TAG_NAME}:{TAG_VERSION} .
# docker run --gpus all -it --name {CONTAINER_NAME} {TAG_NAME}:{TAG_VERSION}

# Dockerfile
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update -y  
RUN apt-get install -y python3-pip python3-dev build-essential  

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility