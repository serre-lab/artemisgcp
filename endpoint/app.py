import logging
import sys
import os
from flask import Flask
from flask import jsonify
from flask import request
import json
import subprocess
from kfp.v2.google.client import AIPlatformClient
from google.cloud import aiplatform as aip



app = Flask(__name__)


@app.route('/prediction', methods=['POST','GET'])
def prediction():

    data = request.json["instances"]

    project_number = os.environ.get("AIP_PROJECT_NUMBER")
    region = os.environ.get("REGION", "")
    pipeline_root = os.environ.get("BUCKET_NAME", "gs://vertex-ai-sdk-pipelines")
    artifact_uri = os.environ.get("AIP_STORAGE_URI", "")
    service_account = os.environ.get("SERVICE_ACCOUNT", "")

    print(project_number)
    print(region)
    print(pipeline_root)
    print(artifact_uri)
    print(service_account)

    aip.init(project = project_number, location = region)
    # api_client = AIPlatformClient(project_id='acbm-317517', region=region)

    # for key in data[0]:

    print(data[0].values())
    job = aip.PipelineJob(
        display_name = "acbm-inference-pipeline",
        enable_caching = False,
        template_path = "inference_pipeline.json",
        parameter_values={
            'video_files': list(data[0].values()),
            'artifact_uri': artifact_uri
        },
        pipeline_root=pipeline_root
    )

    job.run(
        service_account = service_account,
        sync = False
    )
    # response = api_client.create_run_from_job_spec(
    #     'inference_pipeline.json',
    #     pipeline_root=pipeline_root,
    #     enable_caching = False,
    #     service_account = service_account,
    #     parameter_values={
    #         'video_file': data[0][key],
    #         'artifact_uri': artifact_uri
    #     })
    

    return {"predictions": 'success'}

@app.route('/health', methods=['POST','GET'])
def healthcheck():
    """Return status upon google request"""
    return "healthy", 200 

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)