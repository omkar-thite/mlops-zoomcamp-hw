import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    categorical = ['PULocationID', 'DOLocationID']

    actual_transformation = prepare_data(df, categorical)

    expected = df.copy()
    expected['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    expected['duration'] = expected['duration'].dt.total_seconds() / 60
    
    duration_mask = (expected['duration'] >= 1) & (expected['duration'] <= 60)
    expected = expected[duration_mask]    
    
    expected[categorical] = expected[categorical].fillna(-1).astype('int').astype('str')

    # Reset index to match actual output
    expected = expected.reset_index(drop=True)
    actual_transformation = actual_transformation.reset_index(drop=True)

    print(f"expected: {expected}")
    print(f"actual: {actual_transformation}")

    pd.testing.assert_frame_equal(actual_transformation, expected)

if __name__ == "__main__":
     test_prepare_data()