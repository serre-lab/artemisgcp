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

    # response = api_client.create_run_from_job_spec(
    # 'inference_pipeline.json',
    # pipeline_root='gs://vertex-ai-sdk-pipelines',
    # enable_caching = False,
    # parameter_values={
    #     'project_id': 'acbm-317517',
    #     'video_file': 'gs://acbm_videos/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000_tesing.mp4',
    # })


    return {"predictions": 'help'}

@app.route('/health', methods=['POST','GET'])
def healthcheck():
    """Return status upon google request"""
    return "healthy", 200 

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)