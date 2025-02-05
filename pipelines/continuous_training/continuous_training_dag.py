import os
from datetime import datetime

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

from utils.callbacks import failure_callback, success_callback
from utils.common import read_sql_file

local_timezone = pendulum.timezone("Asia/Seoul")
conn_id = "feature_store"
airflow_dags_path = Variable.get("AIRFLOW_DAGS_PATH")
sql_file_path = os.path.join(
    airflow_dags_path,
    "pipelines",
    "continuous_training",
    "data_extract",
    "features.sql",
)

with DAG(
    dag_id="credit_score_classification_ct",
    default_args={
        "owner": "user",
        "depends_on_past": False,
        "email": ["otzslayer@gmail.com"],
        "on_failure_callback": failure_callback,
        "on_success_callback": success_callback,
    },
    description="A DAG for continuous training",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=local_timezone),
    catchup=False,
    tags=["lgcns", "mlops"],
) as dag:
    # data_extract = EmptyOperator(task_id="데이터_추출")

    data_extract = SQLExecuteQueryOperator(
        task_id="데이터_추출",
        conn_id=conn_id,
        sql=read_sql_file(sql_file_path),
        split_statements=True,
    )

    data_preprocessing = EmptyOperator(task_id="데이터_전처리")

    training = EmptyOperator(task_id="모델_학습_및_평가")

    data_extract >> data_preprocessing >> training
