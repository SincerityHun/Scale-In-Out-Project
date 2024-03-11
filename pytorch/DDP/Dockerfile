#Dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt-get update -y  
RUN apt-get install -y python3-pip python3-dev build-essential  
# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the dependencies file to the working directory
COPY requirements.txt ./
COPY  pytorch_DDP.py ./

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run on container start
CMD ["python", "main.py"]
