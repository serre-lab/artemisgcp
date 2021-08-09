###LFS RUNNER
# Git LFS runner to get the models from large file storage
FROM ubuntu AS lfs_runner

ARG ssh_prv_key
ARG ssh_pub_key

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
FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15

RUN apt update

# make directories
RUN mkdir -p preprocess && \
    mkdir -p preprocess/models

WORKDIR ./preprocess

COPY --from=lfs_runner ["./artemisgcp/preprocess", "./"]

# Copy git lfs files
COPY --from=lfs_runner ["./artemisgcp/i3d/models", "./models"]

RUN pip install pickle-mixin && \
    pip install PIMS==0.5 && \
    pip install opencv-python==4.4.0.42 && \
    pip install "dask[complete]"

ENTRYPOINT ["/usr/bin/env"]