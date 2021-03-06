import logging
import sys
import os
from flask import Flask
from flask import jsonify
from flask import request
import json
import subprocess
from kfp.v2.google.client import AIPlatformClient



app = Flask(__name__)


@app.route('/prediction', methods=['POST','GET'])
def prediction():
    api_client = AIPlatformClient(project_id='acbm-317517', region='us-central1')

    data = request.json["instances"]
    parameters = request.json["parameters"]
    bucket_name = parameters["bucket"]
    service_account = parameters["service account"]
    project_id = parameters["project id"]

    

    for key in data[0]:

        response = api_client.create_run_from_job_spec(
            'inference_pipeline.json',
            pipeline_root='gs://vertex-ai-sdk-pipelines',
            enable_caching = False,
            service_account = service_account,
            parameter_values={
                'project_id': project_id,
                'video_file': data[0][key],
                'model_uri': bucket_name
            })
    

    return {"predictions": 'success'}

@app.route('/health', methods=['POST','GET'])
def healthcheck():
    """Return status upon google request"""
    return "healthy", 200 

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)