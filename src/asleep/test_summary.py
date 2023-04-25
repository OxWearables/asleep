
import numpy as np
import pandas as pd
import os
from asleep.sleep_stats import summarize_sleep_stages

sleep_block_path = os.path.join('outputs', 'sleep_block.csv')
prediction_path = os.path.join('outputs', 'predictions.csv')
json_path = os.path.join('outputs', 'results.json')
times_path = os.path.join('outputs', 'times.npy')


sleep_block_df = pd.read_csv(sleep_block_path)


prediction_df = pd.read_csv(prediction_path)
print(sleep_block_df.head())
print(prediction_df.head())
times = np.load(times_path)

summarize_sleep_stages(prediction_df['raw_label'], times,
                       json_path, sleep_block_path=sleep_block_path)
