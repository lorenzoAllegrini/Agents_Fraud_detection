import json
import pandas as pd
import numpy as np

def extract_time_series(grouped, target_col):
    results = {}
    for user_id, group in grouped:
        series = group.set_index('timestamp')[target_col]
        series = series.dropna()
        # Basic Stats
        stats = {
            'mean': series.mean(),
            'variance': series.var(),
            'overall_trend': np.polyfit(range(len(series)), series.values, 1)[0] if len(series) > 1 else 0
        }
        # Rolling mean
        rolling_mean = series.rolling(window=min(3, len(series))).mean()

        results[user_id] = {
            'full_series': series,
            'stats': stats,
            'smoothed': rolling_mean
        }

    return results