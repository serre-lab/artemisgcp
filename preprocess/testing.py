import argparse

def preprocess(video_file: str, model_folder_name: str) -> list:
    import dask.dataframe as dd
    from get_embedding_nopad import run_i3d
    from urllib.parse import urlparse
    import os

    def download_blob(bucket_name, source_blob_name, destination_folder):
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
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
    pickle_files.append(    run_i3d(model_folder_name = model_folder_name, 
                                    video_name = 'videos' + os.sep + source_blob_name,
                                    base_result_dir = 'videos',
                                    exp_name = 'rkakodka')  )

    return pickle_files

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file')
    parser.add_argument('--model_folder_name', default= 'models')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    preprocess(video_file= args.video_file, model_folder_name = args.model_folder_name)