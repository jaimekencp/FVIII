# Start with an NVIDIA CUDA base image
FROM nvidia/cuda:11.2.2-base-ubuntu20.04

# Set non-interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive

# Update system packages and install software properties
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common

# Add Python 3.10 PPA and install Python 3.10 along with build tools
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev gcc

# Install pip for Python 3.10
RUN apt-get install -y wget && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Set python3.10 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create a symlink for pip to ensure it points to the correct version
RUN ln -s /usr/local/bin/pip /usr/bin/pip

ENV HOST 0.0.0.0

# Set Port 8080 to be used by the container
EXPOSE 8080

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the environment variables
ENV PYDEVD_WARN_EVALUATION_TIMEOUT=2000
ENV PYDEVD_UNBLOCK_THREADS_TIMEOUT=10

# Install additional dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libpq-dev build-essential

# Ensure the latest versions of pip and setuptools
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools

# Install pip requirements
COPY Project_setup/requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# Copy only the Scripts to the /app directory
COPY Scripts/ /app/Scripts/

# copy Raw_Data folder to /app directory
COPY Raw_data/ /app/Raw_data/

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["python3.10", "Scripts/main.py"]
