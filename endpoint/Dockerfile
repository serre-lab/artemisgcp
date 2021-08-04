FROM python:3.6-jessie as build_pipeline

#set working directory
WORKDIR /app

#handle installing python dependencies
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

#add folders and files to container
ADD . /endpoint
ADD inference_pipeline.py /app/inference_pipeline.py

RUN python inference_pipeline.py

FROM build_pipeline as deploy_endpoint

#copy json file from last build stage
COPY --from=build_pipeline /app/inference_pipeline.json .

ADD . /endpoint
#set environment variables
#server variables
ENV PORT 8080

#pipeline variables
ENV project_id 'acbm-317517'
ENV region 'US-CENTRAL1'
ENV pipeline_root_path 'gs://vertex-ai-sdk-pipelines'

RUN ls

CMD ["gunicorn", "app:app", "--config=config.py"]