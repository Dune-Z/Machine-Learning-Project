import argparse

import numpy as np

from torch import save
from copy import deepcopy

#### Our prediction models
from models import LogisticClassifier, OrdinalClassifier

#### Our complicated preprocessing pipeline
from preprocess import Preprocessor, HandleSizeMapping, ComputeItemVectors, AugmentData
from preprocess import OneHotEncoder, OrdinalEncoder, TargetEncoder
from preprocess import MeanImputer, MedianImputer, ModeImputer, ConstantImputer
from preprocess import StandardScaler, MinMaxScaler, ConstantScaler
from preprocess import DropColumns, SelectOutputColumns

from utils import fetch_train_data, random_split

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=200)

model = LogisticClassifier()
fit_args = {}

#### Our proposed pipeline (not ideal though)
prep = Preprocessor(pipeline=[
    DropColumns(cols=['user_name', 'review', 'review_summary', 'rating']),
    # Map each item size to "size_bias" (See our report)
    HandleSizeMapping(),
    OrdinalEncoder(cols=['fit', 'item_name', 'cup_size']),  # (necessary)
    MeanImputer(
        cols=['weight', 'height', 'bust_size', 'cup_size']),  # (necessary)
    # Compute item vectors (USING Projected Gradient Descent)
    ComputeItemVectors(),
    DropColumns(cols=['size_scheme', 'size']),
    OneHotEncoder(cols=['size_suffix', 'rented_for', 'body_type']),
    StandardScaler(cols=[
        'weight', 'height', 'bust_size', 'cup_size', 'item_weight',
        'item_height', 'item_bust_size', 'item_cup_size'
    ]),
    MinMaxScaler(cols=['age', 'price', 'usually_wear']),
    # Multiplying by 1e-4 to downscale the effect of these features
    ConstantScaler(cols=[
        'age', 'price', 'usually_wear', 'weight', 'height', 'bust_size',
        'cup_size', 'item_weight', 'item_height', 'item_bust_size',
        'item_cup_size'
    ],
                   value=1e-4),
    TargetEncoder(cols=['brand', 'category', 'size_main'],
                  target_cols=['weight', 'height', 'bust_size', 'cup_size'],
                  name='target_encoder'),
    DropColumns(cols=['brand', 'category', 'size_main']),
    # Append the output of 'target_encoder' to the input of the next transformer
    SelectOutputColumns(target='target_encoder'),
    MeanImputer(cols=['age', 'weight', 'height', 'bust_size', 'cup_size']),
    MedianImputer(cols=['price', 'usually_wear']),
    OneHotEncoder(cols=['item_name']),
    # Data Augmentation (See our report)
    AugmentData(target_cols=['weight', 'height', 'bust_size', 'cup_size'],
                ratio_small=0.2,
                ratio_large=0.15),
])
#### Parameters above have not been fine-tuned. Be careful when using it.

#### Pipeline with best performance (It seems weird, but exceeds out expectation)
prep.pipeline = [
    DropColumns(cols=['user_name', 'review', 'review_summary', 'rating']),
    OrdinalEncoder(cols=['fit', 'item_name', 'cup_size']),
    DropColumns(cols=[
        'brand', 'category', 'size_main', 'size_scheme', 'size_suffix',
        'weight', 'height', 'bust_size', 'body_type', 'rented_for', 'cup_size',
        'price', 'usually_wear', 'age'
    ]),
    OneHotEncoder(cols=['item_name', 'size']),
]


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',
                        type=str,
                        default='../models_',
                        help='Directory to save the model.')
    parser.add_argument('--data',
                        type=str,
                        default='../data/train_data.json',
                        help='Path to the training data.')
    return parser.parse_args()


def main(args):
    """
    Main function.
    """

    # Fetch data from course homepage
    train_df = fetch_train_data(path=args.data)

    # Cleanse data
    train_df = prep.cleanse(train_df, is_train=True)
    train_df.dropna(subset=['fit'], inplace=True)

    # Transform data
    train_df_prep = prep.fit_transform(train_df)

    # Save the preprocessor
    save(prep, f'{args.out_dir}/preprocessor.pt')

    # Get feature matrix and target vector
    X = train_df_prep.drop('fit', axis=1).to_numpy(dtype=np.float16)
    y = train_df_prep['fit'].to_numpy(dtype=np.uint8)

    # To tackle class imbalance, we split the majority class (True to Size) into 3 folds,
    # and train the model on each fold separately. When predicting on the test set,
    # we aggregate the predictions from all 3 trained models and take the majority vote.
    def partial_fit(partition: np.ndarray):
        new_model = deepcopy(model)
        new_model.fit(X[partition], y[partition], **fit_args)
        return new_model

    # Collect and save the models
    models = [partial_fit(part) for part in random_split(y)]
    save(models, f'{args.out_dir}/models.pt')


if __name__ == "__main__":
    main(parse_arguments())
