import pandas as pd
import os
import glob
import tensorflow as tf
from batch_predictor import predict_batch

# Set working directory
os.chdir('../..')

# Declare path variables

production_data = 'ml-100_prod'
prediction_data_dir = f'production/production_data/{production_data}.csv'
PIPELINE_NAME = 'DCN-iterate'
PIPE_DIR = 'pipeline'
SERVING_MODEL_DIR = os.path.join(PIPE_DIR, 'serving_model', PIPELINE_NAME)

# Load prediction dataset and model
df = pd.read_csv(prediction_data_dir)
loaded_model = tf.saved_model.load(max(glob.glob(os.path.join(SERVING_MODEL_DIR, '*/')), key=os.path.getmtime))  # Load newest model from 'serving_model'
model_name = max(glob.glob(os.path.join(SERVING_MODEL_DIR, '*/')), key=os.path.getmtime).split("/")[-2]  # Get model name to assign monitoring data to model

MONITORING = f'production/monitoring/{PIPELINE_NAME}_{model_name}_predictions.csv'

# Add rating columns to be filled by predict_batch
df['pred_rating'] = 0
df['pred_int_rating'] = 0

# Fill in columns for float prediction value and int prediction value
df_pred = predict_batch(df, loaded_model)

# Add predictions to a monitoring table (simple CSV file)
if os.path.exists(MONITORING):
    df_monitor = pd.read_csv(MONITORING)
    df_joint = pd.concat([df_monitor, df_pred], ignore_index=True)
    df_joint.to_csv(MONITORING, index=False)
else:
    df_pred.to_csv(MONITORING, index=False)