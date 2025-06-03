import pickle
import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error 

import mlflow 

# Mlflow server
mlflow.set_tracking_uri('http://localhost:5000/') 
mlflow.set_experiment('nyc-taxi-experiment')

os.makedirs('models', exist_ok=True)

def read_dataframe(year=None, month=None, **kwargs):
    
    # Get input either passed through cmd with --conf or through UI
    if year==None and month==None:
        
        conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
        year = year or conf.get('year')  
        month = month or conf.get('month')

    if not year or not month:
        raise ValueError("year and month must be provided either as args or in DAG conf")
    
    #url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

    print('Downloading data...')
    df = pd.read_parquet(url)  
    print('Download Complete')

    print(f"df.shape: {df.shape}.")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime   # parquet files have datetime objects, no need to convert
    
    # Convert datetime to minutes
    df.duration = df.duration.apply(lambda x: x.total_seconds()/60.0)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    #categorical = ['VendorID', 'store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    # df['PO_DO'] = df['PULocationID'] + '_' + df['DOLocationID']  comment for homework
    print(f"df.shape after processing: {df.shape}.")
    print(f'year/month: {year}/{month}')
    
    return df


def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True) 
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    y = df['duration'].values

    return X, y, dv

def train_model(X_train, y_train):
    with mlflow.start_run():
        mlflow.sklearn.autolog()

        run_id = mlflow.active_run().info.run_id

        model = LinearRegression()

        model.fit(X_train, y_train)
        
        print('Model intercept:', model.intercept_)
        mlflow.sklearn.log_model(model, artifact_path='mlartifacts')
        mlflow.log_metric('model_intercept', model.intercept_)


if __name__=="__main__":
    df = read_dataframe(year=2023, month=3)
    print(df.columns)