import os
import pandas as pd
from datetime import datetime 

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df = pd.DataFrame(data, columns=columns)

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")

options = {
    'client_kwargs': {
        'endpoint_url': S3_ENDPOINT_URL
    }
}

input_file = 's3://nyc-duration/2023-01.parquet'

df.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

# Run batch.py for January 2023
os.system("export INPUT_FILE_PATTERN='s3://nyc-duration/{year:04d}-{month:02d}.parquet'")
os.system("python batch.py 2023 1")

year = 2023
month = 1

output_file = f's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
df_result = pd.read_parquet(output_file, storage_options=options)

# Calculate sum of predicted durations
sum_predictions = df_result['predicted_duration'].sum()
print(f"Sum of predicted durations: {sum_predictions}")