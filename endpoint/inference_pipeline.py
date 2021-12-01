import kfp
from kfp.v2 import compiler
from google.cloud import aiplatform
import kfp.components as comp
from kfp.v2.google.client import AIPlatformClient
from kfp.components import OutputPath
import google.cloud.aiplatform as aip
# from kfp.dsl import Output
import os

project_id = 'acbm-317517'
region = 'us-central1'
pipeline_root = 'gs://vertex-ai-sdk-pipelines'
pipeline_root_path = pipeline_root
service_account = "vertex-ai-pipeline@acbm-317517.iam.gserviceaccount.com" # @param {type: "string"}


download_model_component = comp.load_component_from_text("""
name: Download model
description: Download model given an artifact URI

inputs:
- {name: model_folder, type: String, description: 'Storage URI where model is saved'}

outputs:
- {name: model_artifact, type: Artifact, description: 'Pytorch model file'}

implementation:
    container:
        image: 'gcr.io/acbm-317517/download_model:dev'
        command: [
            python,
            download_model.py,
            --artifact_uri,
            {inputValue: model_folder},
            --model_file,
            {outputPath: model_artifact},
        ]
""")

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
    image: gcr.io/acbm-317517/preprocess:dev
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
- {name: model_file, type: Artifact, description: 'Model uri path'}

outputs:
- {name: Predictions, type: Artifact, description: 'Predictions of pipeline'}

implementation:
  container:
    image: gcr.io/acbm-317517/inference:dev
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python, 
      # Path of the program inside the container
      dset_main_inf_forshed.py,
      --embname,
      {inputPath: embeddings},
      --model_uri,
      {inputPath: model_file},
      --output_file_name,
      {outputPath: Predictions}
    ]""")

def upload_predictions(video_file: str, predictions: comp.InputArtifact()):
    from google.cloud import storage

    def parse_url(url: str):
        from urllib.parse import urlparse
        o = urlparse(url)
        return o.netloc, o.path.lstrip('/')

    def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"
        # The path to your file to upload
        # source_file_name = "local/path/to/file"
        # The ID of your GCS object
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

    bucket_name, source_blob_name = parse_url(video_file)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_blob_name = video_file.replace('.mp4', '.txt')

    upload_blob(bucket_name, predictions, destination_blob_name)

upload_component = comp.create_component_from_func(
    upload_predictions,
    base_image = 'gcr.io/acbm-317517/utils:dev'
)

@kfp.dsl.pipeline(
    name="acbm-inference-pipeline",
    pipeline_root=pipeline_root_path)
def pipeline(video_files: str, artifact_uri: str):
    
    with kfp.dsl.ParallelFor(video_files) as video:
        preprocess_op = (preprocess_component(
            video_uri = video,
            model_uri= 'models/',
        ).add_node_selector_constraint(
            'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
        ).set_gpu_limit(1))

        download_model_op = download_model_component(
            model_folder = artifact_uri
        )

        inference_op = (inference_component(
            embeddings = preprocess_op.output,
            model_file = download_model_op.output
        ).add_node_selector_constraint(
            'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
        ).set_gpu_limit(1))

        upload_predictions_op = upload_component(
            video_file = video,
            predictions = inference_op.output
        )

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='inference_pipeline.json')

# aip.init(project = "82539680728", location = "us-central1")

# job = aip.PipelineJob(
#     display_name = "acbm-inference-pipeline",
#     enable_caching = False,
#     template_path = "inference_pipeline.json",
#     parameter_values={
#         'video_files': ["gs://test_pipeline_2/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000.mp4", "gs://test_pipeline_2/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000.mp4"],
#         'artifact_uri': "gs://test_pipeline_2/2021_11_16_10_04_07"
#     },
#     pipeline_root=pipeline_root
# )

# job.run(
#     service_account = service_account,
#     sync = False
# )

# api_client = AIPlatformClient(project_id=project_id, region=region)

# response = api_client.create_run_from_job_spec(
#     'inference_pipeline.json',
#     pipeline_root="gs://test_pipeline_2",
#     enable_caching = False,
#     service_account = 'vertex-ai-pipeline@acbm-317517.iam.gserviceaccount.com',
#     parameter_values={
#         'project_id': project_id,
#         'video_file': 'gs://acbm_videos/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000_tesing.mp4',
#         'model_uri': 'acbm_videos'
#     })

# api_client = AIPlatformClient(project_id='acbm-317517', region=region)

# api_client.create_run_from_job_spec(
#             'inference_pipeline.json',
#             pipeline_root="gs://test_pipeline_2",
#             enable_caching = False,
#             service_account = service_account,
#             parameter_values={
#                 'video_file': "gs://test_pipeline_2/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000.mp4",
#                 'artifact_uri': 'gs://example_training_data/2021_11_11_13_23_36'
#             })