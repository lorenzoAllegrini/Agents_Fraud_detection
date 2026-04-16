import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

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

def load_data_for_user():
    pass

def link_user_to_transactions(user: Dict[str, Any], transactions_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Links a user to their transaction history by matching their IBAN against sender and recipient records.

    This function iterates through a transaction dataset to extract only the financial 
    activities associated with a specific user. It checks both the 'Sender IBAN' and 
    'Recipient IBAN' fields to ensure a complete financial footprint is retrieved, 
    regardless of whether the user initiated or received the funds.

    Args:
        user: A dictionary containing user metadata. Must include an 'iban' key.
        transactions_dataset: A list of dictionaries, where each dictionary represents 
            a transaction and contains 'Sender IBAN' and 'Recipient IBAN' keys.

    Returns:
        A list of transaction dictionaries linked to the user's IBAN. Returns an 
        empty list if the user has no IBAN or no matching transactions are found.
    """
    user_iban = user.get('iban')
    
    if not user_iban:
        return []

    user_transactions = [
        tx for tx in transactions_dataset
        if tx.get('Sender IBAN') == user_iban or tx.get('Recipient IBAN') == user_iban
    ]
        
    return user_transactions

def link_user_to_sms(user: Dict[str, Any], sms_dataset: List[Dict[str, str]], confidence_threshold: int = 1) -> List[Dict[str, str]]:
    """Links a user to their corresponding SMS history using heuristic text matching.

    This function evaluates each SMS in the provided dataset against the user's 
    metadata (first name, last name, and city of residence). A confidence score 
    is calculated based on the presence of these attributes within the message 
    text using exact word boundary matching. Messages meeting or exceeding the 
    specified confidence threshold are returned.

    Args:
        user: A dictionary containing user metadata. Expected keys include 
            'first_name', 'last_name', and a nested 'residence' dictionary 
            with a 'city' key.
        sms_dataset: A list of dictionaries, where each dictionary represents 
            an SMS and contains an 'sms' key with the message text.
        confidence_threshold: The minimum score required for an SMS to be 
            associated with the user. Defaults to 1.

    Returns:
        A list of SMS dictionaries that meet the confidence threshold.
    """
    user_sms_history = []
    
    first_name = user.get('first_name', '').lower()
    last_name = user.get('last_name', '').lower()
    city = user.get('residence', {}).get('city', '').lower()
    
    if not first_name:
        return []

    for sms_entry in sms_dataset:
        message_text = sms_entry.get('sms', '').lower()
        score = 0
        
        if first_name and re.search(rf'\b{re.escape(first_name)}\b', message_text):
            score += 1
            
        if last_name and re.search(rf'\b{re.escape(last_name)}\b', message_text):
             score += 1

        if city and re.search(rf'\b{re.escape(city)}\b', message_text):
            score += 1
            
        if score >= confidence_threshold:
            user_sms_history.append(sms_entry)
            
    return user_sms_history



# for the location analysis
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def extract_spatial_timeseries(df, lat_col='lat', lon_col='lng', time_col='timestamp'):
    """
    Extract spatial timseries for the locations analysis.
    Requires df to be 'groupby' with 'biotag'.
    """
    results = {}
    for biotag, group in df:
        # We shift the lat/lon to get the "previous" point's coordinates
        lats = group[lat_col].values
        lons = group[lon_col].values
        
        # Calculate distance between point i and point i-1
        distances = [0] # First point has 0 distance traveled
        for i in range(1, len(group)):
            d = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
            distances.append(d)
        
        group['step_distance'] = distances
        
        # 3. Time Delta (for velocity)
        # Convert nanoseconds to hours for km/h
        time_diffs = group[time_col].diff().dt.total_seconds() / 3600
        group['velocity'] = group['step_distance'] / time_diffs
        
        # 4. Spatial Metrics
        results[biotag] = {
            'total_dist': group['step_distance'].sum(),
            'avg_velocity': group['velocity'].replace([np.inf, -np.inf], np.nan).mean(),
            'max_velocity': group['velocity'].max(),
            # Centroid (Mean location)
            'center': (group[lat_col].mean(), group[lon_col].mean()),
            # Radius of Gyration (how far the user typically wanders from their center)
            'spatial_variance': np.sqrt(np.mean(
                haversine_distance(group[lat_col].mean(), group[lon_col].mean(), lats, lons)**2
            ))
        }
        
    return results