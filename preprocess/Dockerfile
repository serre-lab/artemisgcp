FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15

RUN apt update

RUN mkdir -p preprocess

WORKDIR ./preprocess

COPY ["./preprocess/central_reservoir", "./preprocess/get_embedding_nopad.py", "./preprocess/read_videos.py", "./preprocess/requirements.txt", "./"]

COPY ["./i3d/models", "./"]

RUN pip install pickle-mixin && \
    pip install PIMS==0.5 && \
    pip install opencv-python==4.4.0.42 && \
    pip install "dask[complete]"

RUN ls -ltr

ENTRYPOINT ["/usr/bin/env"]