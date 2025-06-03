from datetime import datetime
import tempfile
import joblib
import os

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from duration_prediction import read_dataframe, create_X, train_model

# GCP VM config
XCOM_BASE_DIR = os.path.expanduser('/mnt/disks/data-disk/airflow/xcoms')

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
}

def ingest_data_callable(**context):
    
    # Get year and month from context
    conf = context.get('dag_run').conf if context.get('dag_run') else {}
    year = conf.get('year', 2023)  # set default year 2023
    month = conf.get('month', 3)    # set default month 3

    if not year or not month:
        raise ValueError("year and month must be provided either as args or in DAG conf")
    
    # Create train and val dataframes
    
    df = read_dataframe(year, month)

    run_id = context['run_id'] if 'run_id' in context else context['dag_run'].run_id
    task_id = context['ti'].task_id
    
    os.makedirs(XCOM_BASE_DIR, exist_ok=True)
    tmp_dir = os.path.join(XCOM_BASE_DIR, f"{run_id}_{task_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    

    joblib.dump(df, os.path.join(tmp_dir, "df.pkl"))
    
    # Push file paths to XCom
    context['ti'].xcom_push(key='data_dir', value=tmp_dir)



def create_dataset_callable(**context):
        
    # Pull dataframe pushed by ingest_data_callable
    data_dir = context['ti'].xcom_pull(task_ids='ingest_data', key='data_dir')
    df = joblib.load(os.path.join(data_dir, "df.pkl"))
    
    X, y, dv = create_X(df)

    joblib.dump(X, os.path.join(data_dir, "X_train.pkl"))
    joblib.dump(y, os.path.join(data_dir, "y_train.pkl"))
    joblib.dump(dv, os.path.join(data_dir, "dv.pkl"))

    # Push file paths to XCom
    context['ti'].xcom_push(key='data_dir', value=data_dir)



def train_xgb_model_callable(**context):
    
    # Pull X_train, y_train, X_val, y_val from context (Pushed by create_dataset_callable)
    tmp_dir = context['ti'].xcom_pull(key='data_dir', task_ids='create_dataset')

    X_train = joblib.load(os.path.join(tmp_dir, "X_train.pkl"))
    y_train = joblib.load(os.path.join(tmp_dir, "y_train.pkl"))

    # Train the model
    train_model(X_train, y_train)



with DAG(
        'homework',
         default_args=default_args,
         schedule=None, #@daily for daily runs
         catchup=False
) as dag:

    ingest_data = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data_callable,
    )

    create_dataset = PythonOperator(
        task_id='create_dataset',
        python_callable=create_dataset_callable,
    )

    train_xgb_model = PythonOperator(
        task_id='train_model',
        python_callable=train_xgb_model_callable
    )


    ingest_data >> create_dataset >> train_xgb_model 