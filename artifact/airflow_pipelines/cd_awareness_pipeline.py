from airflow import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
from datetime import datetime
import os

# switch to production directory
os.chdir("/home/cory/PycharmProjects/bachelor_2022/artifact/production/monitoring")
print(os.getcwd())

# import functions
from cd_eval_functions.rmse_calculator import rmse_calc
from cd_eval_functions.cd_detector import cd_detector
from cd_eval_functions.rmse_plot_generator import plot_gen as rpg

# Set up Airflow
with DAG("cd_evaluation",  # Dag id
    start_date=datetime(2022, 8, 3),  # start date, the 1st of January 2021
    schedule_interval=None,  # This pipeline should only be externally triggered (e.g. by MLOps Pipeline)
    catchup=False  # Catchup
) as dag:

    def evaluate():
        df = pd.read_csv('DCN-iterate_1658684509_predictions.csv')  # Load prediction score dataset

        df['date'] = pd.to_datetime(df['timestamp'], unit='s')

        # Group by date and calculate the rmse for each month/week/year & join them to one dataset
        group_by = "Y"

        rmse_float = rmse_calc(df[['date', 'user_rating', 'pred_rating']], group_by)  # RMSE of native predictions
        rmse_int = rmse_calc(df[['date', 'user_rating', 'pred_int_rating']], group_by)  # RMSE of rounded predictions
        rmse_df = pd.DataFrame(dict(rmse_float=rmse_float, rmse_int=rmse_int))

        # CD detection (determine whether CD occurred in the data)
        no_cd = cd_detector(rmse_df, delta_threshold=0.1, absolute_threshold=1.5)

        # Write file of the CD evaluation result (prerequisite to CD adaptation: File can be read by Airflow an initiate TFX retraining)
        if no_cd[0]:
            with open('evaluation_outputs/OK.txt', 'w') as file:
                file.write("run:\t\t" + str(datetime.now()) + "\nNO CD DETECTED")
        else:
            with open('evaluation_outputs/CD_DETECTED.txt', 'w') as file:
                file.write("run:\t\t{}\ndelta:\t\t{}\nabsolute:\t{}".format(datetime.now(), no_cd[1][0], no_cd[1][1]))

        # CD understanding (give visual feedback about CD)
        group_by = "M"  # Group by month for better visualization

        vis_rmse_float = rmse_calc(df[['date', 'user_rating', 'pred_rating']], group_by)  # RMSE of native predictions
        vis_rmse_int = rmse_calc(df[['date', 'user_rating', 'pred_int_rating']],
                                 group_by)  # RMSE of rounded predictions
        vis_rmse_df = pd.DataFrame(dict(rmse_float=vis_rmse_float, rmse_int=vis_rmse_int))

        rpg(vis_rmse_df, "evaluation_outputs/rmse_trend.png")

        return not no_cd[0]  # Is passed on to ShortCircuitOperator: If True (CD has been detected) trigger TFX training pipeline

    cd_evaluation = PythonOperator(
        task_id='cd_evaluation',
        python_callable=evaluate,
        dag=dag
    )
