import logging
import os
from flask import Flask
from flask import jsonify
from flask import request
import json
from kfp.v2.google.client import AIPlatformClient


app = Flask(__name__)

project_id = 'acbm-317517'
region = 'us-central1'
pipeline_root_path = 'gs://vertex-ai-sdk-pipelines'
api_client = AIPlatformClient(project_id=project_id, region=region)

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    data = request.json()
    logging.basicConfig(level=logging.INFO)
    logging.info('The video file isrequest body is: {}'.format(data))
    return {"predictions": [data]}

@app.route('/health', methods=['POST','GET'])
def healthcheck():
    """Return status upon google request"""
    return "healthy", 200 

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)