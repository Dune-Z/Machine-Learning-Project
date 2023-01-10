import numpy as np
import pandas as pd
from copy import deepcopy


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


def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   index='result') -> pd.DataFrame:
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
    score_df['#small'] = [np.sum(y_pred == 0).item()]
    score_df['#true2size'] = [np.sum(y_pred == 1).item()]
    score_df['#large'] = [np.sum(y_pred == 2).item()]
    score_df.index = [index]
    return score_df


def train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split dataframe into train and test set
    """
    train_df = df.sample(frac=1 - test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    test_df['fit'].replace({
        'Small': 1,
        'True to Size': 2,
        'Large': 3
    },
                           inplace=True)
    pos = test_df['item_name'].str.contains('\ufeff', na=False)
    test_df.loc[pos, 'fit'] = 2
    return train_df, test_df


def random_split(y: np.ndarray, n_split=3):
    """
    Split data labeled with 'True to Size' into n_split partition.
    :Return: [group_1_index, ..., group_n_split_index]
    """
    small_large_index = np.concatenate(
        [np.where(y == 0)[0], np.where(y == 2)[0]])
    true2size_index = np.where(y == 1)[0]
    np.random.shuffle(true2size_index)
    partitions = np.array_split(true2size_index, n_split)
    partitions = [
        np.concatenate([part, small_large_index]) for part in partitions
    ]
    return partitions


def random_split_aggr(model,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      fit_args: dict = {},
                      n_split=3):
    """
    Apply random split to deal with imbalanced data.
    Require model to have method: fit(X_train, y_train) and predict(X_test)
    Other args are required to be in form of dictionary.
    """

    def partial_fit(partition: np.ndarray):
        print(np.unique(y_train[partition], return_counts=True))
        new_model = deepcopy(model)
        new_model.fit(X_train[partition], y_train[partition], **fit_args)
        return new_model

    partitions = random_split(y_train, n_split=n_split)
    models = [partial_fit(part) for part in partitions]
    predictions = [model.predict(X_test) for model in models]
    predictions = list(map(list, zip(*predictions)))  # list transpose
    aggregate = np.array(
        [max(set(votes), key=votes.count) for votes in predictions])
    return evaluate_model(y_test, aggregate)


def data_augmentation(df: pd.DataFrame,
                      target_cols: list,
                      ratio_small=3.6,
                      ratio_large=2.7):
    """
    Using AFTER 'fit' column is encoded as 0, 1, 2\n
    - For numeric column, random interpolate values between current and extreme value
    - For ordered category column, using extreme case (!!!Warning, may cause overfitting)
    """
    df_small_aug = None
    df_large_aug = None

    df_small = df.groupby('fit').get_group(0)
    if ratio_small <= 1:
        df_small_aug = df_small.sample(frac=ratio_small)
    else:
        df_small_aug = pd.concat([
            df_small.sample(frac=ratio_small - int(ratio_small)),
            pd.concat([df_small for _ in range(int(ratio_small))])
        ],
                                 ignore_index=True)
    for col in target_cols:
        if df[col].dtype == (float or int):
            df_small_aug[col] += np.random.rand(len(df_small_aug)) * (
                df_small_aug[col].max() - df_small_aug[col])
        elif df[col].dtype.ordered:
            df_small_aug[col] = df[col].max()

    df_large = df.groupby('fit').get_group(2)
    if ratio_large <= 1:
        df_large_aug = df_large.sample(frac=ratio_large)
    else:
        df_large_aug = pd.concat([
            df_large.sample(frac=ratio_large - int(ratio_large)),
            pd.concat([df_large for _ in range(int(ratio_large))])
        ],
                                 ignore_index=True)
    for col in target_cols:
        if df[col].dtype == (float or int):
            df_large_aug[col] -= np.random.rand(len(df_large_aug)) * (
                df_large_aug[col] - df_large_aug[col].min())
        elif df[col].dtype.ordered:
            df_large_aug[col] = df[col].min()

    return pd.concat([df, df_small_aug, df_large_aug], ignore_index=True)
