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
    output = subprocess.check_output("ls /credentials", shell=True)
    print(output, flush=True)
    return jsonify("yes")

@app.route('/health', methods=['POST','GET'])
def healthcheck():
    """Return status upon google request"""
    return "healthy", 200 

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)



"""
{
"instances" : [
{
 "help": "help"
}
  ]

}
"""