FROM python:3.6-jessie AS pipeline_runner

RUN apt update

RUN pip install kfp && \
    pip install google-cloud-aiplatform

COPY ["./inference_pipeline.py", "./"]

RUN python inference_pipeline.py

FROM python:3.6-jessie

#set working directory
WORKDIR ./app

#handle installing python dependencies
COPY ./requirements.txt /app/requirements.txt

#add folders and files to container
COPY ["./app.py", "./config.py", "./"]

COPY --from=pipeline_runner ["./inference_pipeline.json", "./"]

#set environment variables
#server variables
ENV PORT 8080

#pipeline variables
ENV project_id 'acbm-317517'
ENV region 'US-CENTRAL1'
ENV pipeline_root_path 'gs://vertex-ai-sdk-pipelines'

RUN pip install -r requirements.txt
# RUN python inference_pipeline.py

CMD ["gunicorn", "app:app", "--config=config.py"]
