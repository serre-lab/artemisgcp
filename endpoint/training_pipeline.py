import kfp
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from kubernetes.client.models import V1EnvVar
import kfp.components as comp

project_id = 'acbm-317517'
region = 'US-CENTRAL1'
pipeline_root_path = 'gs://vertex-ai-sdk-pipelines'

#KFP pipeline. Needs name and root path where artifacts stored
@kfp.dsl.pipeline(
    name="automl-image-inference-v2",
    pipeline_root=pipeline_root_path)

#Preprocess component. Input: video_uri - Output: embeddings
#def preprocess(video_file: str) -> str:
#    import os
#    import logging
#    logging.basicConfig(level=logging.INFO)
#    logging.info('The video file is: {}'.format(video_file))

#    return video_file + '.p'

#Training component. Inputs: annotations_uri, embeddings_uri - Ouput: confusion matrix (not there yet but end goal)
def traing(video_file: str, embeddings: str):

    import os
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info('The video file is: {}, The embeddings file is : {}'.format(video_file, embeddings))

#preprocess_step = comp.create_component_from_func(
#    preprocess,
#    base_image='gcr.io/deeplearning-platform-release/tf-gpu.1-15'
#    )

train_step = comp.create_component_from_func(
    train,
    base_image='gcr.io/deeplearning-platform-release/tf-gpu.1-15'
    )


def pipeline(project_id: str, video_file: str):

    #preprocess_op = (preprocess_step(video_file).
    #    add_node_selector_constraint('cloud.google.com/gke-accelerator', 'nvidia-tesla-k80').
    #    set_gpu_limit(1))
    inference_op = train_step(video_file, preprocess_op.output)

compiler.Compiler().compile(pipeline_func=pipeline,
        package_path='inference_pipeline.json')
