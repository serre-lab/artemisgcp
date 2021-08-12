###LFS RUNNER
# Git LFS runner to get the models from large file storage
FROM ubuntu AS lfs_runner

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install git-lfs && \
    git lfs install

RUN git clone https://github.com/serre-lab/artemisgcp.git

RUN ls -ltr

# download models from lfs
RUN cd artemisgcp && \
    git lfs pull -I i3d/models

### MAIN CONTAINER
FROM tensorflow/tensorflow:1.15.5-gpu

RUN apt update && \
    apt-get install ffmpeg libsm6 libxext6  -y

# make directories
RUN mkdir -p preprocess && \
    mkdir -p preprocess/models && \
    mkdir -p videos

WORKDIR ./preprocess

COPY ["./preprocess", "./"]

# Copy git lfs files
COPY --from=lfs_runner ["./artemisgcp/i3d/models", "./models"]

RUN pip install pickle-mixin && \
    pip install PIMS==0.5 && \
    pip install opencv-python==4.4.0.42 && \
    pip install "dask[complete]" && \
    pip install google-cloud-storage

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
ENV PYTHONPATH /preprocess:$PYTHONPATH

ENTRYPOINT ["/bin/bash"]