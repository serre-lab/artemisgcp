from google.cloud import storage
import json
from ast import literal_eval
from urllib.parse import urlparse
import subprocess

from google.cloud.storage import blob


client = storage.Client()

annotation_bucket = client.bucket('acbm_videos')

blobs = annotation_bucket.list_blobs()

annotationBlob = annotation_bucket.blob('test/testset_GT')

data = annotationBlob.download_as_bytes()


print('help', data)
for blob in blobs:
  
    print(blob.name)


