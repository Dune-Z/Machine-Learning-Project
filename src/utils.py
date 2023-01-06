import numpy as np
import pandas as pd


def fetch_train_data(
    path='../data/train_data.json',
    url='http://miralab.ai/courses/ML2022Fall/project/train_data_all.json'
) -> pd.DataFrame:
    import os, requests
    from tqdm import tqdm
    if not os.path.exists(path):
        print(f'Downloading {path} from {url}')
        response = requests.get(url, stream=True)
        length = int(response.headers.get('content-length', 0))
        with open(path, 'wb') as file:
            iterator = response.iter_content(chunk_size=1024)
            iterator = tqdm(iterator, total=length // 1024, unit='KB')
            for data in iterator:
                file.write(data)
    return pd.read_json(path)


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


def onehot_encoding(df:pd.DataFrame, columns:list, change_columns=True):
    """
    Onehot encoding using `pd.get_dummies`\n
    May be used in columns with relative less categories otherwise the dataframe would be sparse
    """
    if change_columns:
        df = pd.get_dummies(data=df, columns=columns)
    else:
        # Replacing entries with one-hot vector
        # columns in dataframe will not be changed
        # TODO
        pass
    return df


def label_encoding(df:pd.DataFrame, columns:list):
    """
    Converting category into number (str -> float)\n
    The smaller the frequency is, the larger the encoded number is\n
    Using `fillna(0)` to convert nan into 0
    """
    for column in columns:
        value_dict = df[column].value_counts().keys()
        mapping = dict()
        for (key, value) in enumerate(value_dict):
            mapping[value] = key + 1
        mapping
        df[column].replace(mapping, inplace=True)
        df[column] = df[column].astype(float, copy=False)
        df[column].fillna(0, inplace=True)
    return df


def ordinal_encoding(df:pd.DataFrame, column='fit', order=['Small', 'True to Size', 'Large']):
    """
    Ordinal encoding, only implemented to 'fit' column
    """
    mapping = dict()
    for (index,label) in enumerate(order):
        mapping[label] = index
    df[column].replace(mapping, inplace=True)
    return df


def normalize_column(df:pd.DataFrame, columns:list, method='std', fill_na=0):
    """
    Normalize numeric column, providing methods: std (standard) and minmax\n
    Replacing nan with fill_na, default 0
    """
    if method == 'std':
        for column in columns:
            df[column].fillna(fill_na, inplace=True)
            df[column] = (df[column] - df[column].mean()) / df[column].std()

    if method == 'minmax':
        for column in columns:
            df[column].fillna(fill_na, inplace=True)
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        
    return df

