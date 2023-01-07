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
    partitions = [
        np.concatenate([part, small_large_index]) for part in partitions
    ]
    return partitions


def random_split_aggr(model, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, fit_args: dict):
    """
    Apply random split to deal with imbalanced data.
    Require model to have method: fit(X_train, y_train) and predict(X_test)
    Other args are required to be in form of dictionary.
    """

    def partial_fit(X_train_: np.ndarray, y_train_: np.ndarray):
        model.fit(X_train_, y_train_, **fit_args)
        return model

    partitions = random_split(y_train)
    models = [partial_fit(X_train[part], y_train[part]) for part in partitions]
    predictions = [model.predict(X_test) for model in models]
    predictions = list(map(list, zip(*predictions)))  # list transpose
    aggregate = [max(set(votes), key=votes.count) for votes in predictions]
    evaluate_model(y_test, aggregate)
