from google.cloud import storage
import json
from ast import literal_eval
from urllib.parse import urlparse
import subprocess

from google.cloud.storage import blob


client = storage.Client()

annotation_bucket = client.bucket('acbm_videos')

blobs = annotation_bucket.list_blobs()

for blob in blobs:
  
    if '.json' in blob.name:
        print(blob.name)

count = len(open('mydata.json').readlines(  ))

print(count)