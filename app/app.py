import logging
import os
from flask import Flask


app = Flask(__name__)


@app.route('/prediction')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.route('/health')
def healthcheck():
    """Return status upon google request"""
    return "healthy", 200 

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)