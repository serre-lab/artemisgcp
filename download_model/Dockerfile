###LFS RUNNER
# Git LFS runner to get the models from large file storage
FROM ubuntu AS lfs_runner

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install git-lfs && \
    git lfs install

RUN git clone -b feat-add-pyyaml-model-ranking https://github.com/serre-lab/artemisgcp.git

RUN ls -ltr

# download models from lfs
RUN cd artemisgcp && \
    git lfs pull -I LSTM/models
    
## download model base container
FROM python:3.5

RUN mkdir -p download_model && \
    mkdir -p download_model/models

WORKDIR ./download_model

COPY ["./download_model/download_model.py", "./"]
# Copy git lfs files
COPY --from=lfs_runner ["./artemisgcp/LSTM/models/base_model.pth", "./artemisgcp/LSTM/models/models.yaml", "./models"]

RUN pip install google-cloud-storage

ENTRYPOINT ["bin/bash"]