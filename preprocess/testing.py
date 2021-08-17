import argparse
from genericpath import exists

def preprocess(video_file: str, model_folder_name: str, output_folder: str) -> list:
    import dask.dataframe as dd
    from get_embedding_nopad import run_i3d
    from urllib.parse import urlparse
    from pathlib import Path
    import os

    def download_blob(bucket_name, source_blob_name, destination_folder):
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        print(destination_folder + os.sep + source_blob_name)
        blob.download_to_filename(destination_folder + os.sep + source_blob_name)

        print(
            "Blob {} downloaded to {}.".format(
                source_blob_name, destination_folder + os.sep + source_blob_name
            )
        )

    def parse_url(url: str):
        o = urlparse(url)
        return o.netloc, o.path.lstrip('/')

    pickle_files = []

    bucket_name, source_blob_name = parse_url(video_file)
    download_blob(bucket_name, source_blob_name, 'videos')
    dirname = os.path.dirname(output_folder)
    Path(dirname).mkdir(parents= True, exist_ok=True)
    run_i3d(model_folder_name = model_folder_name, 
            video_name = 'videos' + os.sep + source_blob_name,
            batch_size = 32,
            first_how_many= 108000,
            base_result_dir = (output_folder),
            exp_name = 'rkakodka')

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file')
    parser.add_argument('--model_folder_name', default= 'models')
    parser.add_argument('--output_folder', default = 'videos')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    preprocess(video_file= args.video_file, model_folder_name = args.model_folder_name, output_folder = args.output_folder)