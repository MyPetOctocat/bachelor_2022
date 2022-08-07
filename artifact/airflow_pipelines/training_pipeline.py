
from tfx import v1 as tfx
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig


from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline

import os
#from datetime import datetime
import datetime

import tensorflow_model_analysis as tfma
from typing import List

os.chdir("/home/cory/PycharmProjects/bachelor_2022/artifact") # change to working_dir

PIPELINE_NAME = 'DCN-airflow'
#WORKING_DIR = 'bachelor_2022/artifact'
PIPE_DIR = 'pipeline'

# Directory where MovieLens 100K rating data resides
DATA_ROOT = os.path.join('data', PIPELINE_NAME)
print(DATA_ROOT)
# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = os.path.join(PIPE_DIR, 'pipelines', PIPELINE_NAME)
print(PIPELINE_ROOT)
# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join(PIPE_DIR, 'metadata', PIPELINE_NAME, 'metadata.db')
print(METADATA_PATH)
# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join(PIPE_DIR, 'serving_model', PIPELINE_NAME)
print(SERVING_MODEL_DIR)

MODEL_PLOTS = os.path.join(PIPE_DIR, 'pipelines', PIPELINE_NAME, 'plots')
print(MODEL_PLOTS)

from absl import logging
logging.set_verbosity(logging.INFO)  # Set default logging level.

_trainer_module_file = 'model_source/dcn_ranking_training.py'

_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    '--direct_num_workers=0',
]

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2022, 8, 3),
}



def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str, plot_path: str, beam_pipeline_args: List[str]) -> tfx.dsl.Pipeline:


  # Brings data into the pipeline.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Evaluate rudimentary statistics on the dataset
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples']
      )

  # Read and infer schema of the dataset
  schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics']
  )

  # Check dataset for anomalies
  example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      train_args=tfx.proto.TrainArgs(num_steps=12),
      eval_args=tfx.proto.EvalArgs(num_steps=24),
      custom_config={"plot_path": plot_path})

  # Pushes the model to a filesystem destination.
  pusher = tfx.components.Pusher(
      model=trainer.outputs['model'],
      push_destination=tfx.proto.PushDestination(
          filesystem=tfx.proto.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  # Following three components will be included in the pipeline.
  components = [
      example_gen,
      statistics_gen,
      schema_gen,
      example_validator,
      trainer,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      components=components,
      beam_pipeline_args=beam_pipeline_args
  )

DAG = AirflowDagRunner((AirflowPipelineConfig(_airflow_config))).run(
  _create_pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      data_root=DATA_ROOT,
      module_file=_trainer_module_file,
      serving_model_dir=SERVING_MODEL_DIR,
      metadata_path=METADATA_PATH,
      plot_path=MODEL_PLOTS,
      beam_pipeline_args=_beam_pipeline_args))
