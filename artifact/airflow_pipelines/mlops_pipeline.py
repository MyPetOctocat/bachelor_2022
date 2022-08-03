from airflow import DAG
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from cd_awareness_pipeline import evaluate

with DAG("a_mlops_pipeline",  # Dag id
    start_date=datetime(2022, 8, 3),  # start date, the 1st of January 2021
    schedule_interval='@daily',  # Cron expression, here it is a preset of Airflow, @daily means once every day.
    catchup=False  # Catchup
) as dag:

    cd_eval = ShortCircuitOperator(
        task_id='cd_evaluation',
        python_callable=evaluate,
        dag=dag)

    train = TriggerDagRunOperator(
        task_id='retrain',
        trigger_dag_id='DCN-airflow',
        dag=dag
    )

    cd_eval >> train
