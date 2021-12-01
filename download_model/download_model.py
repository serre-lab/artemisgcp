def download_model(artifact_uri: str, model_file: str):
    import os
    from pathlib import Path
    
    def download_blob(bucket_name, source_blob_name, destination_file_name):
        from google.cloud import storage
        """Downloads a blob from the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"

        # The ID of your GCS object
        # source_blob_name = "storage-object-name"

        # The path to which the file should be downloaded
        # destination_file_name = "local/path/to/file"

        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
                source_blob_name, bucket_name, destination_file_name
            )
        )

    def parse_url(url: str):
        from urllib.parse import urlparse
        o = urlparse(url)
        return o.netloc, o.path.lstrip('/')

    def get_model_path(yaml_file):
        import yaml
        with open(yaml_file, 'r') as f:
            document = yaml.safe_load(f)
        model_path = document['path']
        return model_path

    bucket_name, source_blob_name = parse_url(artifact_uri)

    download_blob(bucket_name, os.path.join(source_blob_name, 'model.yaml'), 'models/model.yaml')
    model_path = get_model_path('models/model.yaml')

    dirname = os.path.dirname(model_file)
    Path(dirname).mkdir(parents= True, exist_ok=True)

    download_blob(bucket_name, os.path.join(source_blob_name, model_path), model_file)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact_uri', type=str, required= True)
    parser.add_argument('--model_file', type=str, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    download_model(artifact_uri=args.artifact_uri, model_file=args.model_file)
    