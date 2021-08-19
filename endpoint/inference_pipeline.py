import kfp
from kfp.v2 import compiler
from google.cloud import aiplatform
import kfp.components as comp
# from kfp.dsl import Output
import os

project_id = 'acbm-317517'
region = 'US-CENTRAL1'
pipeline_root_path = 'gs://vertex-ai-sdk-pipelines'

# def preprocess(video_file: str, model_folder_name: str) -> list:
#     import dask.dataframe as dd
#     from get_embedding_nopad import run_i3d
#     from urllib.parse import urlparse
#     import os

#     def download_blob(bucket_name, source_blob_name, destination_folder):
#         from google.cloud import storage
#         storage_client = storage.Client()
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(source_blob_name)
#         print(destination_folder + os.sep + source_blob_name)
#         blob.download_to_filename(destination_folder + os.sep + source_blob_name)

#         print(
#             "Blob {} downloaded to {}.".format(
#                 source_blob_name, destination_folder + os.sep + source_blob_name
#             )
#         )

#     def parse_url(url: str):
#         o = urlparse(url)
#         return o.netloc, o.path.lstrip('/')

#     pickle_files = []

#     bucket_name, source_blob_name = parse_url(video_file)
#     print(bucket_name, source_blob_name)
#     download_blob(bucket_name, source_blob_name, '/preprocess/videos')
#     pickle_files.append(    run_i3d(model_folder_name = model_folder_name, 
#                                     video_name = '/preprocess/videos' + os.sep + source_blob_name,
#                                     base_result_dir = 'videos',
#                                     exp_name = 'rkakodka')  )

#     return pickle_files

preprocess_component = comp.load_component_from_text("""
name: Get embeddings
description: Run the i3d model to get embeddings

inputs:
- {name: video_uri, type: String, description: 'URI of the video file (GCP bucket)'}
- {name: model_uri, type: String, description: 'URI to Annotations to use for training'}

outputs:
- {name: pickled_output, type: Artifact, description: 'URI to Saved model to use for predictions'}

implementation:
  container:
    image: gcr.io/acbm-317517/i3d-preprocess:latest
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python, 
      # Path of the program inside the container
      testing.py,
      --video_file,
      {inputValue: video_uri},
      --model_folder_name, 
      {inputValue: model_uri},
      --output_folder,
      {outputPath: pickled_output},
    ]""")

inference_component = comp.load_component_from_text("""
name: Run LSTM model
description: Runs the LSTM model on embeddings

inputs:
- {name: embeddings, type: Artifact, description: 'Path for embeddings file'}

outputs:
- {name: Predictions, type: Artifact, description: 'Predictions of pipeline'}

implementation:
  container:
    image: gcr.io/acbm-317517/lstm-inference:latest
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python, 
      # Path of the program inside the container
      dset_main_inf_forshed.py,
      --embname,
      {inputPath: embeddings},
      --output_file_name,
      {outputPath: Predictions}
    ]""")

# def inference(embeddings: kfp.dsl.io_types.Output[kfp.dsl.io_types.Artifact()]):

#     import os
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     logging.info('The embeddings file is : {}'.format(embeddings))

@kfp.dsl.pipeline(
    name="automl-image-inference-v2",
    pipeline_root=pipeline_root_path)
def pipeline(project_id: str, video_file: str):

    # preprocess_step = comp.create_component_from_func(
    #     preprocess,
    #     base_image='gcr.io/acbm-317517/i3d-preprocess'
    #     )

    preprocess_op = (preprocess_component(
        video_uri = video_file,
        model_uri= 'models/',
    ).add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
    ).set_gpu_limit(1))

    inference_op = (inference_component(
        embeddings = preprocess_op.output,
    ).add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
    ).set_gpu_limit(1))

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='inference_pipeline.json')
