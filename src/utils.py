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


def evaluate_model(y_true, y_pred, index='result') -> pd.DataFrame:
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
    score_df.index = [index]
    return score_df


def onehot_encoding(df: pd.DataFrame, columns: list, change_columns=True):
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


def label_encoding(df: pd.DataFrame, columns: list):
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
        df[column].replace(mapping, inplace=True)
        df[column] = df[column].astype(float, copy=False)
        df[column].fillna(0, inplace=True)
    return df


def ordinal_encoding(df: pd.DataFrame,
                     column='fit',
                     order=['Small', 'True to Size', 'Large']):
    """
    Ordinal encoding, only implemented to 'fit' column
    """
    mapping = dict()
    for index, label in enumerate(order):
        mapping[label] = index
    df[column].replace(mapping, inplace=True)
    return df


def normalize_column(df: pd.DataFrame, columns: list, method='std', fill_na=0):
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
            df[column] = (df[column] - df[column].min()) / (df[column].max() -
                                                            df[column].min())

    return df


def train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split dataframe into train and test set
    """
    train_df = df.sample(frac=1 - test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    return train_df, test_df


def random_split(y: np.ndarray, n_split=5):
    """
    Split data labeled with 'True to Size' into n_split partition.
    :Return: [group_1_index, ..., group_n_split_index]
    """
    small_large_index = np.where(y == 1 or y == 3)[0]
    true2size_index = np.where(y == 2)[0]
    np.random.shuffle(true2size_index)
    partitions = np.array_split(true2size_index, n_split)
    partitions = [np.concatenate([part, small_large_index]) for part in partitions]
    return partitions


def random_split_aggr(model,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray
                      ):
    def partial_fit(X_train_: np.ndarray, y_train_: np.ndarray):
        model.fit(X_train_, y_train_)
        return model

    partitions = random_split(y_train)
    models = [partial_fit(X_train[part], y_train[part]) for part in partitions]
    predictions = [model.predict(X_test) for model in models]
    predictions = list(map(list, zip(*predictions)))  # list transpose
    aggregate = [max(set(votes), key=votes.count) for votes in predictions]
    evaluate_model(y_test, aggregate)
