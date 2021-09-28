import subprocess
    
from google.cloud import storage
from urllib.parse import urlparse
import re

def download_model(source_blob_model):
    client = storage.Client()
    model_accuracy = None
    

    model_exists = False
    for model in client.list_blobs('test_pipeline_1', prefix='trained_models'):
        res = re.findall("\d+\.\d+", model.name)
        print(model.name)
        model_exists = True
        
        if model_accuracy == None:
            
            model_accuracy= float(res[0])
            model_name = model.name
        
        else:
            if model_accuracy > float(res[0]):
                continue
            else:
                model_accuracy = float(res[0])
                model_name = model.name

    print('ran through model')
    print(model_exists)
    if model_exists == True:     
        model_bucket = client.bucket(source_blob_model)
        modelBlob = model_bucket.blob(model_name)
        print('model found')
        modelBlob.download_to_filename("try.pth")
    if model_exists == False:
        print("no model downloading one")
        model_file = "models/"

if __name__ == "__main__":
    download_model('test_pipeline_1')