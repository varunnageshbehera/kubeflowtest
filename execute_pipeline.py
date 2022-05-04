import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import component

client = kfp.Client('http://10.20.3.244:30900')

######################## run the pipeline function #####################

#client.create_run_from_pipeline_func(add_pipeline, arguments={'a': 7, 'b': 8})


"""
client.create_run_from_pipeline_func(
    add_pipeline,
    arguments={'a': 7, 'b': 8},
    mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
)
"""

##################### load yaml file and run the pipeline ##################################
client.create_run_from_pipeline_package(pipeline_file ='/home/teamai/pipeline.yaml',arguments={'a': 10, 'b': 2})

### refer https://kubeflow-pipelines.readthedocs.io/en/latest/source/kfp.client.html ####

