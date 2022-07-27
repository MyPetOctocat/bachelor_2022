
from typing import Dict, Text
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import tensorflow as tf

from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_recommenders as tfrs
from tensorflow_transform.tf_metadata import schema_utils
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

_FEATURE_KEYS = ["movie_id","user_id","user_gender", "user_occupation", "user_age_cohort"]
_LABEL_KEY = 'user_rating'

_FEATURE_SPEC = {
    **{
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
        for feature in _FEATURE_KEYS
    }, _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
}

class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    # Define the dimension the feature values should be embedded in
    embedding_dimension = 32
    self.embedding_dims = embedding_dimension
    # Create np array with incrementing values as the vocabulary
    unique_user_ids = np.array(range(943)).astype(str)
    unique_movie_ids = np.array(range(1682)).astype(str)
    unique_occupation_ids = np.array(range(21)).astype(str)
    unique_gender_ids = np.array(range(2)).astype(str)
    unique_age_ids = np.array(range(7)).astype(str)


    ## String values embeddings
    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), name='user_id', dtype=tf.int64),
        tf.keras.layers.Lambda(lambda x: tf.as_string(x)),
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        # Create embedding layer of dimension 943x32
        tf.keras.layers.Embedding(
            len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), name='movie_id', dtype=tf.int64),
        tf.keras.layers.Lambda(lambda x: tf.as_string(x)),
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_ids, mask_token=None),
        tf.keras.layers.Embedding(
            len(unique_movie_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for occupations.
    self.occupation_embeddings = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), name='user_occupation', dtype=tf.int64),
        tf.keras.layers.Lambda(lambda x: tf.as_string(x)),
        tf.keras.layers.StringLookup(
            vocabulary=unique_occupation_ids, mask_token=None),
        tf.keras.layers.Embedding(
            len(unique_occupation_ids) + 1, embedding_dimension)
    ])

    ## Int value embeddings
    # Compute embeddings for gender.
    self.gender_embeddings = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), name='user_gender', dtype=tf.int64),
        tf.keras.layers.IntegerLookup(
            vocabulary=unique_gender_ids, mask_token=None),
        tf.keras.layers.Embedding(
            len(unique_gender_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for age.
    self.age_embeddings = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,), name='user_age_cohort', dtype=tf.int64),
        tf.keras.layers.IntegerLookup(
            vocabulary=unique_age_ids, mask_token=None),
        tf.keras.layers.Embedding(
            len(unique_age_ids) + 1, embedding_dimension)
    ])

    # Cross Layer
    self.cross_layer = tfrs.layers.dcn.Cross(kernel_initializer=tf.keras.initializers.RandomNormal(seed=1)) # Use seeds to make model reproducible

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
        self.cross_layer,
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(seed=1)),
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(seed=1)),
        tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(seed=1))
    ])

  def call(self, inputs):

    user_id, movie_id, user_gender, user_occupation, user_age = inputs

    # Calculate embedding for each feature and save in *_embedding variable
    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_id)
    gender_embedding = self.gender_embeddings(user_gender)
    occupation_embedding = self.occupation_embeddings(user_occupation)
    age_embedding = self.age_embeddings(user_age)


    # Create embedding layer
    return self.ratings(tf.concat([user_embedding, movie_embedding, gender_embedding, occupation_embedding, age_embedding], axis=2))


class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()])

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model((features['user_id'], features['movie_id'], features['user_gender'], features['user_occupation'], features['user_age_cohort']))

  def compute_loss(self,
                   features: Dict[Text, tf.Tensor],
                   training=False) -> tf.Tensor:

    labels = features[1]
    rating_predictions = self(features[0])

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 256) -> tf.data.Dataset:
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      schema=schema).repeat()


def _build_keras_model() -> tf.keras.Model:
  return MovielensModel()


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  # Generate training logfiles for tensorboard
  from datetime import datetime
  logdir = "pipeline/pipelines/DCN-iterate/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

  # Derive data schema from generated _FEATURE_SPEC dictionary
  schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

  train_dataset = _input_fn(
      fn_args.train_files, fn_args.data_accessor, schema, batch_size=8192)
  eval_dataset = _input_fn(
      fn_args.eval_files, fn_args.data_accessor, schema, batch_size=4096)

  model = _build_keras_model()

  model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      epochs = 200,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  model.save(fn_args.serving_model_dir)


  ###  Display model summary
  print("\n#####################################")
  print(model.summary())
  print()

  # Save plot of model architecture
  model_num = fn_args.serving_model_dir.split("/")[-2]   # extract model number
  img_dir = fn_args.custom_config["plot_path"] + f"/{model_num}"
  print(img_dir)
  Path(img_dir).mkdir(parents=True, exist_ok=True)
  tf.keras.utils.plot_model(model.ranking_model.ratings, to_file=f"{img_dir}/model_architecture_{model_num}.png", show_shapes=True)
  print()

  ### Cross feature Visualization
  mat = model.ranking_model.cross_layer._dense.kernel # Cross weights matrix
  features = _FEATURE_KEYS

  block_norm = np.ones([len(features), len(features)])
  dim = model.ranking_model.embedding_dims

  # Compute the norms of the blocks.
  for i in range(len(features)):
    for j in range(len(features)):
      # Norm of 32x32 Matrix is calculated | 32x32 values --> 1 value
      block = mat[i * dim:(i + 1) * dim,    # 32x32 blocks are retrieved from cross network
                  j * dim:(j + 1) * dim]
      block_norm[i,j] = np.linalg.norm(block, ord="fro") # Frobenius norm is used | norm of each matrix element is calculated and added together
  # Create plot
  plt.figure(figsize=(9,9))
  im = plt.matshow(block_norm, cmap=plt.cm.Blues)
  ax = plt.gca()
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(im, cax=cax)
  cax.tick_params(labelsize=10)
  _ = ax.set_xticklabels([""] + features, rotation=45, ha="left", fontsize=10)
  _ = ax.set_yticklabels([""] + features, fontsize=10)

  plt.savefig(f"{img_dir}/cross_features_{model_num}", dpi=500, bbox_inches='tight')
