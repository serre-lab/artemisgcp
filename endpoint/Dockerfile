FROM python:3.6-jessie

RUN apt update

#set working directory
WORKDIR /app

#handle installing python dependencies
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

#add folders and files to container
ADD . /app
ADD inference_pipeline.py /app/inference_pipeline.py

#set environment variables
#server variables
ENV PORT 8080

#pipeline variables
ENV project_id 'acbm-317517'
ENV region 'US-CENTRAL1'
ENV pipeline_root_path 'gs://vertex-ai-sdk-pipelines'


RUN python inference_pipeline.py

RUN ls /app

