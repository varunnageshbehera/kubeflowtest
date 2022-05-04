import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import component

@component
def add(a: float, b: float) -> float:
  '''Calculates sum of two arguments'''
  return a + b

@component
def multiply(a: float, b: float) -> float:
  '''Calculates sum of two arguments'''
  return a * b



@dsl.pipeline(
  name='add_pipeline',
  description='An example pipeline that performs calculations.',
  # pipeline_root='gs://my-pipeline-root/example-pipeline'
)

def add_pipeline(a: float=1, b: float=7):
  res = add(a, b)
  multiply(res.output, b)


from kfp import compiler
compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(pipeline_func=add_pipeline, package_path='pipeline.yaml')
#compiler.Compiler().compile(pipeline_func=add_pipeline, package_path='pipeline3.yaml')

client = kfp.Client('http://10.20.3.244:30900')
# run the pipeline in v2 compatibility mode
#client.create_run_from_pipeline_func(add_pipeline, arguments={'a': 7, 'b': 8})

client.create_run_from_pipeline_func(
    add_pipeline,
    arguments={'a': 7, 'b': 8},
    mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
)


