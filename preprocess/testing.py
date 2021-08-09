import argparse

def preprocess(video_file: str) -> list:
    import dask.dataframe as dd
    from get_embedding_nopad import run_i3d
    from urllib.parse import urlparse

    def download_blob(bucket_name, source_blob_name, destination_file_name):
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        print(
            "Blob {} downloaded to {}.".format(
                source_blob_name, destination_file_name
            )
        )

    def parse_url(url: str):
        o = urlparse(url)
        return o.netloc, o.path

    model_folder_name = "models"

    pickle_files = []

    for videos in video_file:
        bucket_name, source_blob_name = parse_url(videos)
        download_blob(bucket_name, source_blob_name, 'videos')
        pickle_files.append(    run_i3d(model_folder_name = model_folder_name, 
                                        video_name = 'videos',
                                        base_result_dir = 'videos',
                                        exp_name = 'rkakodka')  )

    return pickle_files

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('video_file')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    preprocess(args.video_file)