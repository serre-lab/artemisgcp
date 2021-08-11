FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-1

# make directories
RUN mkdir -p training && \
    mkdir -p training/central_reservoir && \
    mkdir -p training/models

# set working directory
WORKDIR ./training

# COPY FILES INTO CONTAINER

# local files
COPY ["./acbm-317517-7d3150de2508.json", "./"]
COPY ["./training/utils.py", "./training/dataset_load_torch.py", "./training/requirements.txt", "./training/training_runjob.sh", "./training/main_training.py", "./training/baseline.py", "./"]

# git lfs files
#RUN curl -O https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh &&\
#    bash script.deb.sh

COPY ["./LSTM/models", "./models"]
#RUN apt-get install git-lfs && \
#    git-lfs install
#RUN cd models && \
#    git-lfs pull -I *

# install dependencies
RUN pip install -r /training/requirements.txt


# command after every docker run
#ENTRYPOINT ["/usr/bin/env", "/training/training_runjob.sh"]

#dummy CMD that is overwritten by argument in docker run (docker run -i "video string")
#CMD [""]



#ARG GOOGLE_APPLICATION_CREDENTIALS


#ENV GOOGLE_APPLICATION_CREDENTIALS=./training/acbm-317517-7d3150de2508.json

# # Install gcsfuse.
#RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
#     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#     apt-get update && \
#     apt-get install -y gcsfuse

#RUN gcsfuse acbm_videos ./videos

#RUN curl -O https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh &&\
 #    bash script.deb.sh

#uncomment when pushing to github actions
#RUN cd models && \
 #   git-lfs pull --include "LSTM/models/*"

