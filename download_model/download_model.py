
import yaml


def get_best_model(raw_txt):
    import yaml

    document = yaml.safe_load(raw_txt)

    models = document['models']

    accuracy = 0
    best_model = {}
    for version in models:
        if (models[version]['accuracy'] > accuracy):
            accuracy = models[version]['accuracy']
            best_model = models[version]

    return best_model['path']

def download_best_model(bucket_name: str, model_file: str):

    from google.cloud import storage

    client = storage.Client()

    model_bucket = client.bucket(bucket_name)
    yaml_blob = model_bucket.blob('trained_models/models.yaml')
    if yaml_blob.exists():
        print("Loading best model ...")
        document = yaml_blob.download_as_text()
        model_path = get_best_model(document)
        model_blob = model_bucket.blob(model_path)
        print("Found the best model as {}".format(model_path))
        model_file = 'best_model.pth'
        model_blob.download_to_filename(model_file)
    else:
        print("Could not find a yaml file, loading base model")
        model_file = 'models/base_model.pth'
        document = 'models/models.yaml'
        model_blob = model_bucket.blob('trained_models/base_model.pth')
        print("Uploading base model to the main bucket")
        yaml_blob.upload_from_filename(document)
        model_blob.upload_from_filename(model_file)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name')
    parser.add_argument('--model_file')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    download_best_model(bucket_name=args.bucket_name, model_file=args.model_file)