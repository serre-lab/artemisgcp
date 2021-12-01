FROM python:3.5

RUN pip install google-cloud-storage && \
    pip install pyyaml

RUN mkdir -p download_model && \
    mkdir -p download_model/models

WORKDIR ./download_model

COPY ["./download_model.py", "./"]

ENTRYPOINT ["bin/bash"]