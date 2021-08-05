FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15

RUN apt update

# make directories
RUN mkdir -p preprocess && \
    mkdir -p preprocess/central_reservoir && \
    mkdir -p preprocess/models && \

# Install gcsfuse.
RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y gcsfuse

# Authenticate and mount gcs bucket
RUN gcloud auth activate-service-account --key-file=${SERVICE_KEY} && \
    gcsfuse acbm_videos ./videos

WORKDIR ./preprocess

COPY ["./preprocess/get_embedding_nopad.py", "./preprocess/read_videos.py", "./preprocess/requirements.txt", "./"]

COPY ["./i3d/models", "./models"]

COPY ["./preprocess/central_reservoir", "./central_reservoir"]

RUN pip install pickle-mixin && \
    pip install PIMS==0.5 && \
    pip install opencv-python==4.4.0.42 && \
    pip install "dask[complete]"

ENTRYPOINT ["/usr/bin/env"]