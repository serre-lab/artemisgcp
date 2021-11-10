from google.cloud import storage
import json
from ast import literal_eval
from urllib.parse import urlparse
import subprocess


client = storage.Client()

def downloadData(annotation_bucket_name='', embedding_bucket_name=''):
    
    annotations_bucket = client.bucket(annotation_bucket_name)
    embeddings_bucket = client.bucket(embedding_bucket_name)

    annotation_list_blobs = annotations_bucket.list_blobs(prefix = "annotations/")


    embedding_list_blobs = embeddings_bucket.list_blobs(prefix = "embeddings/")
    
    for annotation in annotation_list_blobs:
        if annotation.name.endswith(".json"):
            destination_uri = '{}'.format(annotation.name) 
            annotation.download_to_filename(destination_uri)

    for emb in embedding_list_blobs:
        if emb.name.endswith(".p") :
            destination_uri = '{}'.format(emb.name) 
            emb.download_to_filename(destination_uri)

    

  






