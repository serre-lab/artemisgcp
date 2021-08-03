FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15

RUN apt update

WORKDIR ./

COPY ["central_reservoir", "models", "get_embedding_nopad.py", "read_videos", "requirements", "./"]

RUN pip install pickle-mixin && \
    pip install PIMS==0.5 && \
    pip install opencv-python==4.4.0.42 && \
    pip install "dask[complete]"