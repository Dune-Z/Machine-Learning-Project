import numpy as np
import pandas as pd


def fetch_train_data(
    path='../data/train_data.json',
    url='http://miralab.ai/courses/ML2022Fall/project/train_data_all.json'
) -> pd.DataFrame:
    import os, requests, tqdm
    if os.path.exists(path):
        return pd.read_json(path)
    print(f'Downloading {path} from {url}')
    response = requests.get(url, stream=True)
    length = int(response.headers.get('content-length', 0))
    with open(path, 'wb') as file:
        iterator = response.iter_content(chunk_size=1024)
        iterator = tqdm(iterator, total=length // 1024, unit='KB')
        for data in iterator:
            file.write(data)


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    desc_df = pd.DataFrame()
    desc_df['dtype'] = df.dtypes
    desc_df['valid_count'] = df.count()
    desc_df['nan_count'] = df.isnull().sum()
    desc_df['unique_count'] = df.nunique()
    return desc_df


def evaluate_model(y_true, y_pred) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    score_df = pd.DataFrame()
    score_df['accuracy'] = [accuracy_score(y_true, y_pred)]
    score_df['precision'] = [
        precision_score(y_true, y_pred, average='macro', zero_division=0)
    ]
    score_df['recall'] = [
        recall_score(y_true, y_pred, average='macro', zero_division=0)
    ]
    score_df['f1'] = [
        f1_score(y_true, y_pred, average='macro', zero_division=0)
    ]
    score_df['f1_weighted'] = [
        f1_score(y_true, y_pred, average='weighted', zero_division=0)
    ]
    return score_df
