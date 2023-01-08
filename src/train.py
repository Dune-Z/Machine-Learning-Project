import argparse

import numpy as np
import pandas as pd

from torch import save, load
from copy import deepcopy

from utils import fetch_train_data, random_split
from models import *
from preprocess import *


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",
                        type=str,
                        default="../models",
                        help="Directory to save the model.")
    parser.add_argument("--data",
                        type=str,
                        default="../data/train_data_all_filled.json",
                        help="Path to the training data.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size for training.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=100,
                        help="Number of epochs to train the model.")
    return parser.parse_args()


def main(args):
    """
    Main function.
    """
    prep = Preprocessor()
    model = OrdinalClassifier()
    fit_args = {}

    # Fetch data from course homepage
    train_df = fetch_train_data(path=args.data)

    # Cleanse data
    train_df = prep.cleanse(train_df, is_train=True)
    train_df.dropna(subset=['fit'], inplace=True)

    # Transform data
    prep.pipeline = [
        DropColumns(cols=[
            'user_name', 'review', 'review_summary', 'rating', 'item_name'
        ]),
        OneHotEncoder(cols=[
            'size_scheme', 'size_main', 'size_suffix', 'rented_for',
            'body_type'
        ],
                      name='one_hot'),
        OrdinalEncoder(cols=['fit', 'cup_size']),
        StandardScaler(
            cols=['age', 'weight', 'height', 'bust_size', 'cup_size']),
        TargetEncoder(
            cols=['brand', 'category', 'size'],
            target_cols=['weight', 'height', 'bust_size', 'cup_size'],
            name='target_encoder'),
        DropColumns(cols=['brand', 'category', 'size']),
        MinMaxScaler(cols=['price', 'usually_wear']),
        SelectOutputColumns(
            target='target_encoder'
        ),  # append the output of 'one_hot' to the input of the next transformer
        MeanImputer(cols=['age', 'weight', 'height', 'bust_size', 'cup_size']),
        MedianImputer(cols=['usually_wear']),
    ]

    train_df_prep = prep.fit_transform(train_df)
    # Save the preprocessor
    save(prep, args.out_dir + '/preprocessor.pt')

    # Get feature matrix and target vector
    X = train_df_prep.drop('fit', axis=1).values
    y = train_df_prep['fit'].values

    # To tackle class imbalance, we split the majority class (True to Size) into 5 folds,
    # and train the model on each fold separately. When predicting on the test set,
    # we combine the predictions from all 5 folds and take the mode.
    def partial_fit(partition: np.ndarray):
        new_model = deepcopy(model)
        new_model.fit(X[partition], y[partition], **fit_args)
        return new_model

    # Collect and save the models
    models = [partial_fit(part) for part in random_split(y)]
    save(models, args.out_dir + '/models.pt')


if __name__ == "__main__":
    main(parse_arguments())
