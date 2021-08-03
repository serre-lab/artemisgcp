FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15

RUN apt update

RUN mkdir -p preprocess && \
    mkdir -p preprocess/central_reservoir && \
    mkdir -p preprocess/models

WORKDIR ./preprocess

COPY ["./preprocess/get_embedding_nopad.py", "./preprocess/read_videos.py", "./preprocess/requirements.txt", "./"]

COPY ["./i3d/models", "./models"]

COPY ["./preprocess/central_reservoir", "./central_reservoir"]

RUN pip install pickle-mixin && \
    pip install PIMS==0.5 && \
    pip install opencv-python==4.4.0.42 && \
    pip install "dask[complete]"

RUN ls -ltr models

ENTRYPOINT ["/usr/bin/env"]