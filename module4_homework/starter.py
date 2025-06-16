import os
import sys
import pickle

import pandas as pd 

print("Loading model...")

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

print("Parsing arguments...")
try:
    year = int(sys.argv[1])
    month = int(sys.argv[2])
except (ValueError, TypeError):
    raise ValueError(f"Invalid format for year or month.")


print(f"Reading data for {month} {year}...")
df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)

print("Applying model...")
y_pred = model.predict(X_val)

std = y_pred.std()
mean = y_pred.mean()

print(f'Standard deviation for March 2023 duration predictions: {std}')
print(f"Mean predicted duration for March 2023: {mean}")

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

print(f"Writing result in DataFrame...")
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predictions'] = y_pred

output_file = 'output/results.parquet'

print(f"Saving results to {output_file}...")

os.makedirs(os.path.dirname(output_file), exist_ok=True)

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

if os.path.exists(output_file):
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Output file size: {file_size:.2f} MB")
else:
    print("Output file not found!")
