FROM python:3.6-jessie

RUN apt update

#set working directory
WORKDIR ./app

#handle installing python dependencies
ADD ./endpoint/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

#add folders and files to container
ADD ./endpoint/app.py /app/app.py
ADD ./endpoint/inference_pipeline.py /app/inference_pipeline.py
ADD ./endpoint/config.py /app/config.py

#set environment variables
#server variables
ENV PORT 8080

#pipeline variables
ENV project_id 'acbm-317517'
ENV region 'US-CENTRAL1'
ENV pipeline_root_path 'gs://vertex-ai-sdk-pipelines'


RUN python inference_pipeline.py


CMD ["gunicorn", "app:app", "--config=config.py"]