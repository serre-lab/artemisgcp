from yaml.events import DocumentEndEvent
import kfp
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kubernetes.client.models import V1EnvVar
import kfp.components as comp
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from google.cloud import aiplatform as aip
from typing import NamedTuple
import json

project_id = 'acbm-317517'
region = 'us-central1'
pipeline_root_path = 'gs://vertex-ai-sdk-pipelines'
endpoint_name = 'testing_model_endpoint_1'

create_step_train = comp.load_component_from_text("""
name: Train Model
description: trains LSTM model
inputs:
- {name: annotation_bucket, type: String, description: 'Path to Annotations to use for training'}
- {name: embedding_bucket, type: String, description: 'Path to Embeddings to use for training'}
- {name: save_bucket, type: String, description: 'Path to trained model to use for training'}

outputs:
- {name: models_bucket, type: Artifact, description: 'Bucket where trained models are saved'}

implementation:
  container:
    image: gcr.io/acbm-317517/training:dev
    # command is a list of strings (command-line arguments). 
    # The YAML language has two syntaxes for lists and you can use either of them. 
    # Here we use the "flow syntax" - comma-separated strings inside square brackets.
    command: [
      python, 
      # Path of the program inside the container
      /training/main_training.py,
      --emb, 
      {inputValue: embedding_bucket},
      --annotation, 
      {inputValue: annotation_bucket},
      --save,
      {inputValue: save_bucket},
      --trained_models_folder,
      {outputPath: models_bucket}
    ]""")

upload_model_component = comp.load_component_from_text("""
name: Upload best model
description: Uploads the best model to a GCP bucket along woth model metadata
inputs:
- {name: model_bucket, type: Artifact, description: 'Path to the bucket where models are stored'}
- {name: best_model_bucket, type: String, description: 'URI where the best model along with its metadata is to be stored'}

outputs:
- {name: best_model, type: Artifact, description: 'Details of best chosen model'}

implementation:
    container: 
        image: 'gcr.io/acbm-317517/upload_best_model:dev'
        command: [
            python,
            upload_best_model.py,
            --bucket_name,
            {inputPath: model_bucket},
            --model_folder,
            {inputValue: best_model_bucket},
            --best_model,
            {outputPath: best_model}
        ]""")

# model_deploy_comp = comp.load_component_from_text("""
# name: model_deploy
# description: |
#     Deploys a Google Cloud Vertex Model to the Endpoint, creating a DeployedModel within it.
#     For more details, see https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/deployModel.
#     Args:
#         model (google.VertexModel):
#             Required. The model to be deployed.
#         endpoint (google.VertexEndpoint):
#             Required. The endpoint to be deployed to.
#         deployed_model_display_name (Optional[str]):
#             The display name of the DeployedModel. If not provided
#             upon creation, the Model's display_name is used.
#         traffic_split (Optional[Dict[str, int]]):
#             A map from a DeployedModel's ID to the percentage
#             of this Endpoint's traffic that should be forwarded to that DeployedModel.
#             If this field is non-empty, then the Endpoint's trafficSplit
#             will be overwritten with it. To refer to the ID of the just
#             being deployed Model, a "0" should be used, and the actual ID
#             of the new DeployedModel will be filled in its place by this method.
#             The traffic percentage values must add up to 100.
#             If this field is empty, then the Endpoint's trafficSplit is not updated.
#         dedicated_resources_machine_type (Optional[str]):
#             The specification of a single machine used by the prediction.
#             This field is required if `automatic_resources_min_replica_count` is not specified.
#             For more details, see https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints#dedicatedresources.
#         dedicated_resources_accelerator_type (Optional[str]):
#             Hardware accelerator type. Must also set accelerator_count if used.
#             See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#AcceleratorType
#             for available options.
#             This field is required if `dedicated_resources_machine_type` is specified.
#         dedicated_resources_accelerator_count (Optional[int]):
#             The number of accelerators to attach to a worker replica.
#         dedicated_resources_min_replica_count (Optional[int]):
#             The minimum number of machine replicas this DeployedModel will be
#             always deployed on. This value must be greater than or equal to 1.
#             If traffic against the DeployedModel increases, it may dynamically be deployed
#             onto more replicas, and as traffic decreases, some of these extra replicas may be freed.
#         dedicated_resources_max_replica_count (Optional[int]):
#             The maximum number of replicas this deployed model may
#             the larger value of min_replica_count or 1 will
#             be used. If value provided is smaller than min_replica_count, it
#             will automatically be increased to be min_replica_count.
#             The maximum number of replicas this deployed model may
#             be deployed on when the traffic against it increases. If requested
#             value is too large, the deployment will error, but if deployment
#             succeeds then the ability to scale the model to that many replicas
#             is guaranteed (barring service outages). If traffic against the
#             deployed model increases beyond what its replicas at maximum may
#             handle, a portion of the traffic will be dropped. If this value
#             is not provided, will use dedicated_resources_min_replica_count as
#             the default value.
#         automatic_resources_min_replica_count (Optional[int]):
#             The minimum number of replicas this DeployedModel
#             will be always deployed on. If traffic against it increases,
#             it may dynamically be deployed onto more replicas up to
#             automatic_resources_max_replica_count, and as traffic decreases,
#             some of these extra replicas may be freed. If the requested value
#             is too large, the deployment will error.
#             This field is required if `dedicated_resources_machine_type` is not specified.
#         automatic_resources_max_replica_count (Optional[int]):
#             The maximum number of replicas this DeployedModel may
#             be deployed on when the traffic against it increases. If the requested
#             value is too large, the deployment will error, but if deployment
#             succeeds then the ability to scale the model to that many replicas
#             is guaranteed (barring service outages). If traffic against the
#             DeployedModel increases beyond what its replicas at maximum may handle,
#             a portion of the traffic will be dropped. If this value is not provided,
#             a no upper bound for scaling under heavy traffic will be assume,
#             though Vertex AI may be unable to scale beyond certain replica number.
#         service_account (Optional[str]):
#             The service account that the DeployedModel's container runs as. Specify the
#             email address of the service account. If this service account is not
#             specified, the container runs as a service account that doesn't have access
#             to the resource project.
#             Users deploying the Model must have the `iam.serviceAccounts.actAs`
#             permission on this service account.
#         disable_container_logging (Optional[bool]):
#             For custom-trained Models and AutoML Tabular Models, the container of the
#             DeployedModel instances will send stderr and stdout streams to Stackdriver
#             Logging by default. Please note that the logs incur cost, which are subject
#             to Cloud Logging pricing.
#             User can disable container logging by setting this flag to true.
#         enable_access_logging (Optional[bool]):
#             These logs are like standard server access logs, containing information like
#             timestamp and latency for each prediction request.
#             Note that Stackdriver logs may incur a cost, especially if your project
#             receives prediction requests at a high queries per second rate (QPS).
#             Estimate your costs before enabling this option.
#         explanation_metadata (Optional[dict]):
#             Metadata describing the Model's input and output for explanation.
#             For more details, see https://cloud.google.com/vertex-ai/docs/reference/rest/v1/ExplanationSpec#explanationmetadata.
#         explanation_parameters (Optional[dict]):
#             Parameters that configure explaining information of the Model's predictions.
#             For more details, see https://cloud.google.com/vertex-ai/docs/reference/rest/v1/ExplanationSpec#explanationmetadata.
#     Returns:
#         gcp_resources (str):
#             Serialized gcp_resources proto tracking the deploy model's long running operation.
#             For more details, see https://github.com/kubeflow/pipelines/blob/master/components/google-cloud/google_cloud_pipeline_components/proto/README.md.
# inputs:
# - {name: model, type: google.VertexModel}
# - {name: endpoint, type: String, optional: true}
# - {name: deployed_model_display_name, type: String, optional: true}
# - {name: traffic_split, type: JsonArray, optional: true, default: '{}'}
# - {name: dedicated_resources_machine_type, type: String, optional: true}
# - {name: dedicated_resources_min_replica_count, type: Integer, optional: true, default: 1}
# - {name: dedicated_resources_max_replica_count, type: Integer, optional: true, default: 1}
# - {name: dedicated_resources_accelerator_type, type: String, optional: true}
# - {name: dedicated_resources_accelerator_count, type: Integer, optional: true, default: 0}
# - {name: automatic_resources_min_replica_count, type: Integer, optional: true, default: 0}
# - {name: automatic_resources_max_replica_count, type: Integer, optional: true, default: 0}
# - {name: service_account, type: String, optional: true}
# - {name: disable_container_logging, type: Boolean, optional: true}
# - {name: enable_access_logging, type: Boolean, optional: true}
# - {name: explanation_metadata, type: JsonObject, optional: true, default: '{}'}
# - {name: explanation_parameters, type: JsonObject, optional: true, default: '{}'}
# outputs:
# - {name: gcp_resources, type: String}
# implementation:
#   container:
#     image: gcr.io/ml-pipeline/google-cloud-pipeline-components:latest
#     command: [python3, -u, -m, google_cloud_pipeline_components.container.experimental.gcp_launcher.launcher]
#     args: [
#       --type, DeployModel,
#       --payload,
#       concat: [
#           '{',
#           '"endpoint": "', {inputValue: endpoint}, '"',
#           ', "traffic_split": ', {inputValue: traffic_split},
#           ', "deployed_model": {',
#           '"model": "', "{{$.inputs.artifacts['model'].metadata['resourceName']}}", '"',
#           ', "dedicated_resources": {',
#           '"machine_spec": {',
#           '"machine_type": "',{inputValue: dedicated_resources_machine_type}, '"',
#           ', "accelerator_type": "',{inputValue: dedicated_resources_accelerator_type}, '"',
#           ', "accelerator_count": ',{inputValue: dedicated_resources_accelerator_count},
#           '}',
#           ', "min_replica_count": ', {inputValue: dedicated_resources_min_replica_count},
#           ', "max_replica_count": ', {inputValue: dedicated_resources_max_replica_count},
#           '}',
#           ', "automatic_resources": {',
#           '"min_replica_count": ',{inputValue: automatic_resources_min_replica_count},
#           ', "max_replica_count": ',{inputValue: automatic_resources_max_replica_count},
#           '}',
#           ', "service_account": "', {inputValue: service_account}, '"',
#           ', "disable_container_logging": "', {inputValue: disable_container_logging}, '"',
#           ', "enable_access_logging": "', {inputValue: enable_access_logging}, '"',
#           ', "explanation_spec": {',
#           '"parameters": ', {inputValue: explanation_parameters},
#           ', "metadata": ', {inputValue: explanation_metadata},
#           '}',
#           '}',
#           '}'
#       ],
#       --project, '', # not being used
#       --location, '', # not being used
#       --gcp_resources, {outputPath: gcp_resources},
#     ]
# """)


def get_accuracy(best_model: InputPath()) -> str:

    import yaml
    import json

    with open(best_model, 'r') as f:
        document = yaml.safe_load(f)
        accuracy = document['accuracy']

    return json.dumps(dict(accuracy = "{}".format(int(accuracy*10000))))


def get_endpoint(project: str, endpoint_name: str, region: str) -> NamedTuple(
    'endpoint',[
        ('exists', str),
        ('endpoint_name', str)
    ]):

    from google.cloud import aiplatform
    from collections import namedtuple
    import json

    aiplatform.init(project=project, location=region)

    endpoint = aiplatform.Endpoint.list(filter = 'display_name={}'.format(endpoint_name))

    if endpoint:
        exists = 'True'
        endpoint_name = endpoint[0].resource_name
    else:
        exists = 'False'
        endpoint_name = 'endpoint not found'

    output = namedtuple(
        'endpoint',
        ['exists', 'endpoint_name']
    )

    return output(exists, endpoint_name)


# ListEndpointsOp = gcc_aip.utils.convert_method_to_component(
#     aiplatform_v1beta1.services.endpoint_service.EndpointServiceClient,
#     aiplatform_v1beta1.services.endpoint_service.EndpointServiceClient.list_endpoints,
# )

# def download_model(source_blob_model: str, model_file: OutputPath()):
#     import subprocess
#     subprocess.run(["pip", "install", "google-cloud-storage"])
#     from google.cloud import storage
#     from urllib.parse import urlparse
#     import re
 
#     client = storage.Client()
#     model_accuracy = None

#     model_exists = False
#     for model in client.list_blobs(source_blob_model, prefix='trained_models/'):
#         if model.name.endswith(".pth"):
#             print(model.name)
#             res = re.findall("\d+\.\d+", model.name)
#             model_exists = True
            
#             if model_accuracy == None:
#                 model_accuracy= float(res[0])
#                 model_name = model.name
            
#             else:
#                 if model_accuracy > float(res[0]):
#                     continue
#                 else:
#                     model_accuracy = float(res[0])
#                     model_name = model.name

#     print('ran through model')
#     print(model_exists)
#     if model_exists == True:     
#         model_bucket = client.bucket(source_blob_model)
#         modelBlob = model_bucket.blob(model_name)
#         print('model found: using model found in ' + source_blob_model + '/' + model_name)
#         modelBlob.download_to_filename(model_file)
#     if model_exists == False:
#         print("no model downloading one")
#         model_file = "models/"

def check_embeddings_exist(video_file: str) -> str:
    ''' Returns exists if a video_files embeddings exists'''
    from google.cloud import storage

    if video_file == 'None':
        return 'Exists'

    def parse_url(url: str):
        from urllib.parse import urlparse
        o = urlparse(url)
        return o.netloc, o.path.lstrip('/')

    bucket_name, source_blob_name = parse_url(video_file)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name.replace('.mp4', '.p').replace('videos/', 'embeddings/'))

    if blob.exists():
        return 'Exists'
    else:
        return 'Does not exist'


def upload_embeddings(video_file: str, embeddings: comp.InputArtifact()):
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
    destination_blob_name = source_blob_name.replace('.mp4', '.p').replace('videos/', 'embeddings/')

    upload_blob(bucket_name, embeddings, destination_blob_name)


get_accuracy_comp = comp.create_component_from_func(
    get_accuracy,
    base_image = 'gcr.io/acbm-317517/utils:dev'
)

get_endpoint_comp = comp.create_component_from_func(
    get_endpoint,
    base_image = 'gcr.io/ml-pipeline/google-cloud-pipeline-components:latest'
)

check_embeddings_component = comp.create_component_from_func(
    check_embeddings_exist, 
    base_image = 'gcr.io/acbm-317517/utils:dev'
)

upload_component = comp.create_component_from_func(
    upload_embeddings,
    base_image = 'gcr.io/acbm-317517/utils:dev'
)

# download_blob_step = comp.create_component_from_func(
#   download_model,
#   base_image='gcr.io/google.com/cloudsdktool/cloud-sdk:latest',
# )

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

#KFP pipeline. Needs name and root path where artifacts stored
@kfp.dsl.pipeline(
    name="automl-image-inference-v2",
    pipeline_root=pipeline_root_path)
def pipeline(project_id: str, model_uri: str, bucket_name: str):
    
    from datetime import datetime
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    with kfp.dsl.ParallelFor(['gs://test_pipeline_2/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000.mp4']) as video:
        check_embeddings_op = check_embeddings_component(video)
        check_embeddings_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        with kfp.dsl.Condition(check_embeddings_op.output != 'Exists'):
            preprocess_op = (preprocess_component(
                video_uri = video,
                model_uri= 'models/',
            ).add_node_selector_constraint(
                'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
            ).set_gpu_limit(1))
            upload_op = upload_component(video, preprocess_op.output)

    
    train_step = create_step_train(
        annotation_bucket=bucket_name,
        embedding_bucket=bucket_name,
        save_bucket=bucket_name,
    ).add_node_selector_constraint(
        'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
    ).set_cpu_limit('4').set_memory_limit('32G').set_gpu_limit(1)

    train_step.after(upload_op)

    upload_best_model_op = (upload_model_component(
      model_bucket=train_step.output,
      best_model_bucket='gs://{}/{}'.format(bucket_name, dt)
    ))

    get_accuracy_op = get_accuracy_comp(
        best_model = upload_best_model_op.output
    )

    model_upload_op = gcc_aip.ModelUploadOp(
      artifact_uri='gs://test_pipeline_2/{}'.format(dt),
      project=project_id,
      display_name='lstm_trained_model',
      serving_container_predict_route='/prediction',
      serving_container_health_route='/health',
      serving_container_image_uri='gcr.io/acbm-317517/endpoint:dev',
      labels=get_accuracy_op.output,
      serving_container_environment_variables=[dict(name = "REGION", value = "us-central1"), dict(name = "BUCKET_NAME", value = "gs://test_pipeline_2"), dict(name = "SERVICE_ACCOUNT", value = "vertex-ai-pipeline@acbm-317517.iam.gserviceaccount.com")]
    )
    model_upload_op.after(upload_best_model_op)

    # get_endpoint_op = get_endpoint_comp(
    #     project = project_id,
    #     endpoint_name = endpoint_name,
    #     region = 'us-central1'
    # )

    # with kfp.dsl.Condition(get_endpoint_op.outputs["exists"] == "False"):
    endpoint_create_op = gcc_aip.EndpointCreateOp(
        project=project_id,
        display_name=endpoint_name,
    )
    #To do: update nedpoint to start inference pipeline.
    model_deploy_op = gcc_aip.ModelDeployOp( 
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=model_upload_op.outputs["model"],
        deployed_model_display_name="lstm_trained_model",
        dedicated_resources_machine_type="n1-standard-4",
        service_account="vertex-ai-pipeline@acbm-317517.iam.gserviceaccount.com",
        dedicated_resources_min_replica_count = 1,
        dedicated_resources_max_replica_count = 1,
        traffic_split = {"0": 100}
    )

    # with kfp.dsl.Condition(get_endpoint_op.outputs["exists"] == 'True'):
    #     model_deploy_op = model_deploy_comp( 
    #         endpoint=get_endpoint_op.outputs["endpoint_name"],
    #         model=model_upload_op.outputs["model"],
    #         deployed_model_display_name="lstm_trained_model",
    #         dedicated_resources_machine_type="n1-standard-4",
    #         service_account="vertex-ai-pipeline@acbm-317517.iam.gserviceaccount.com"
    #     )
    
# def pipeline(project_id: str):
    
#     with kfp.dsl.ParallelFor(
#         ['gs://acbm_videos/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000.mp4', 'gs://acbm_videos/videos/video_2019Y_04M_25D_12h_29m_13s_cam_6394837-0000_tesing.mp4']
#         ) as video:
#         check_embeddings_op = check_embeddings_component(video)
#         check_embeddings_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
#         with kfp.dsl.Condition(check_embeddings_op.output != 'Exists'):
#             preprocess_op = (preprocess_component(
#                 video_uri = video,
#                 model_uri= 'models/',
#             ).add_node_selector_constraint(
#                 'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
#             ).set_gpu_limit(1))
#             upload_op = upload_component(video, preprocess_op.output)

#     print_op = print_component()
#     print_op.after(upload_op)
    

    # check_embeddings_op = check_embeddings_component(
    #     video_file)

    # with kfp.dsl.Condition(check_embeddings_op.output != 'Exists'):
    #     preprocess_op = (preprocess_component(
    #         video_uri = video_file,
    #         model_uri= 'models/',
    #     ).add_node_selector_constraint(
    #         'cloud.google.com/gke-accelerator', 'nvidia-tesla-p100'
    #     ).set_gpu_limit(1))
    

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='training_pipeline.json',
        type_check=False)

# api_client = AIPlatformClient(project_id=project_id, region=region)

# response = api_client.create_run_from_job_spec(
#     'training_pipeline.json',
#     pipeline_root=pipeline_root_path,
#     enable_caching = False,
#     service_account = 'vertex-ai-pipeline@acbm-317517.iam.gserviceaccount.com',
#     parameter_values={
#         'project_id': project_id,
#         'model_uri': 'test_pipeline_2',
#         'bucket_name': 'test_pipeline_2',
#     })

aip.init(project = '82539680728', location = region)

job = aip.PipelineJob(
    display_name = "automl-image-inference-v2",
    enable_caching = False,
    template_path = "training_pipeline.json",
    parameter_values={
        'project_id': project_id,
        'model_uri': 'test_pipeline_2',
        'bucket_name': 'test_pipeline_2',
    },
    pipeline_root=pipeline_root_path
).run(
    service_account = 'vertex-ai-pipeline@acbm-317517.iam.gserviceaccount.com',
    sync = False
)