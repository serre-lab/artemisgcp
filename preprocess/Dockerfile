FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15

RUN apt update

WORKDIR ./

COPY ["./preprocess/central_reservoir", "./i3d/models", "./preprocess/get_embedding_nopad.py", "./preprocess/read_videos", "./preprocess/requirements.txt", "./"]

RUN pip install pickle-mixin && \
    pip install PIMS==0.5 && \
    pip install opencv-python==4.4.0.42 && \
    pip install "dask[complete]"