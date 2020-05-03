from spotlight.interactions import Interactions
import pandas as pd

def get_nowplaying_dataset():
    df = pd.read_csv('~/sbcf_dataset/nowplaying/train_processed.csv')
    return Interactions(user_ids=df['user_ids'], item_ids=df['item_ids'], timestamps=df['timestamps'])

def get_retailrocket_dataset():
    df = pd.read_csv('~/sbcf_dataset/retailrocket/retailrocket_processed.csv')
    return Interactions(user_ids=df['user_ids'], item_ids=df['item_ids'], timestamps=df['timestamps'])