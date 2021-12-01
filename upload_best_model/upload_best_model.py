import yaml

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )



def get_best_model(yaml_file):
    import yaml

    with open(yaml_file, 'r') as f:
        document = yaml.safe_load(f)

    models = document['models']

    accuracy = 0
    best_model = {}
    for version in models:
        print(models[version])
        if (float(models[version]['accuracy']) > accuracy):
            accuracy = float(models[version]['accuracy'])
            best_model = models[version]

    return best_model

# def download_best_model(bucket_name: str, model_file: str):

#     from google.cloud import storage
#     from pathlib import Path
#     import os
#     import shutil

#     client = storage.Client()

#     model_bucket = client.bucket(bucket_name)
#     yaml_blob = model_bucket.blob('trained_models/models.yaml')
#     if yaml_blob.exists():
#         print("Loading best model ...")
#         document = yaml_blob.download_as_text()
#         model_path = get_best_model(document)
#         model_blob = model_bucket.blob(model_path)
#         print("Found the best model as {}".format(model_path))
#         dirname = os.path.dirname(model_file)
#         Path(dirname).mkdir(parents= True, exist_ok=True)
#         model_blob.download_to_filename(model_file)
#     else:
#         print("Could not find a yaml file, loading base model")
#         # model_file = 'models/base_model.pth'
#         document = 'models/models.yaml'
#         model_blob = model_bucket.blob('trained_models/base_model.pth')
#         print("Uploading base model to the main bucket")
#         yaml_blob.upload_from_filename(document)
#         model_blob.upload_from_filename('models/base_model.pth')
#         dirname = os.path.dirname(model_file)
#         Path(dirname).mkdir(parents= True, exist_ok=True)
#         shutil.copyfile('models/base_model.pth', model_file)

def upload_best_model(folder_name: str, model_folder: str, best_model: str):

    from pathlib import Path
    import os
    import shutil

    def parse_url(url: str):
        from urllib.parse import urlparse
        o = urlparse(url)
        return o.netloc, o.path.lstrip('/')

    best_model_meta= get_best_model(os.path.join(folder_name, 'models.yaml'))

    shutil.copyfile(os.path.join(folder_name, best_model_meta['path']), 
                    os.path.join('models', best_model_meta['path']))

    shutil.copyfile(os.path.join(folder_name, best_model_meta['confusion_matrix']), 
                    os.path.join('models', best_model_meta['confusion_matrix']))
                    
    dirname = os.path.dirname(best_model)
    Path(dirname).mkdir(parents = True, exist_ok=True)

    with open(best_model,  'w') as f:
        dump = yaml.dump(best_model_meta)
        f.write(dump)

    bucket_name, source_blob_name = parse_url(model_folder)
    upload_blob(bucket_name, 
                os.path.join('models', best_model_meta['path']), 
                os.path.join(source_blob_name, best_model_meta['path']))
    try:
        upload_blob(bucket_name, 
                    os.path.join('models', best_model_meta['confusion_matrix']), 
                    os.path.join(source_blob_name, best_model_meta['path']))
    except KeyError:
        print("confusion matrix for the model wasn't found")
        pass
    upload_blob(bucket_name, best_model, os.path.join(source_blob_name, 'model.yaml'))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str, required= True)
    parser.add_argument('--model_folder', type=str, required=True)
    parser.add_argument('--best_model', type= str, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    upload_best_model(folder_name=args.bucket_name, model_folder=args.model_folder, best_model=args.best_model)