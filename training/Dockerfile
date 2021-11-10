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
    git lfs pull -I LSTM/models


FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-10

# make directories
RUN mkdir -p training && \
    mkdir -p training/central_reservoir && \
    mkdir -p training/models && \
    mkdir -p training/annotations && \
    mkdir -p training/embeddings

# set working directory
WORKDIR ./training

# COPY FILES INTO CONTAINER

# Copy git lfs files
COPY --from=lfs_runner ["./artemisgcp/LSTM/models", "./models"]

COPY ["./training/utils.py", "./training/dataset_load_torch.py", "./training/requirements.txt", "./training/training_runjob.sh", "./training/main_training.py", "./training/baseline.py", "./training/download_blobs.py", "./training/upload_blob.py", "./"]

COPY ["./LSTM/models", "./models"]

# install dependencies
RUN pip install -r /training/requirements.txt && \
    pip install --upgrade protobuf

ENTRYPOINT ["bin/bash"]

