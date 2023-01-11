import argparse

import numpy as np

from torch import save
from copy import deepcopy

from utils import fetch_train_data, random_split
from models import *
from preprocess import *

model = LogisticClassifier()
fit_args = {}

prep = Preprocessor(
    pipeline=[
        ##
        DropColumns(cols=['user_name', 'review', 'review_summary', 'rating']),
        HandleSizeMapping(),  # handle size mapping
        OrdinalEncoder(cols=['fit', 'item_name', 'cup_size']),  # (necessary)
        MeanImputer(
            cols=['weight', 'height', 'bust_size', 'cup_size']),  # (necessary)
        ComputeItemVectors(),  # compute item vectors
        ##
        DropColumns(cols=['size_scheme', 'size']),
        OneHotEncoder(cols=['size_suffix', 'rented_for', 'body_type']),
        StandardScaler(cols=[
            'weight', 'height', 'bust_size', 'cup_size', 'item_weight',
            'item_height', 'item_bust_size', 'item_cup_size'
        ]),
        MinMaxScaler(cols=['age', 'price', 'usually_wear']),
        ConstantScaler(
            cols=[
                'age', 'price', 'usually_wear', 'weight', 'height',
                'bust_size', 'cup_size', 'item_weight', 'item_height',
                'item_bust_size', 'item_cup_size'
            ],
            value=1e-4
        ),  # Multiplying by 1e-4 to downscale the effect of these features
        TargetEncoder(
            cols=['brand', 'category', 'size_main'],
            target_cols=['weight', 'height', 'bust_size', 'cup_size'],
            name='target_encoder'),
        DropColumns(cols=['brand', 'category', 'size_main']),
        SelectOutputColumns(
            target='target_encoder'
        ),  # append the output of 'target_encoder' to the input of the next transformer
        MeanImputer(cols=['age', 'weight', 'height', 'bust_size', 'cup_size']),
        MedianImputer(cols=['price', 'usually_wear']),
        OneHotEncoder(cols=['item_name']),
        AugmentData(target_cols=['weight', 'height', 'bust_size', 'cup_size'],
                    ratio_small=0.2,
                    ratio_large=0.15),
    ])


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',
                        type=str,
                        default='../models',
                        help='Directory to save the model.')
    parser.add_argument('--data',
                        type=str,
                        default='../data/train_data_sample.json',
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
    y = train_df_prep['fit'].to_numpy(dtype=np.float16)

    # To tackle class imbalance, we split the majority class (True to Size) into 3 folds,
    # and train the model on each fold separately. When predicting on the test set,
    # we aggregate the predictions from all 3 folds and vote.
    def partial_fit(partition: np.ndarray):
        new_model = deepcopy(model)
        new_model.fit(X[partition], y[partition], **fit_args)
        return new_model

    # Collect and save the models
    models = [partial_fit(part) for part in random_split(y)]
    save(models, f'{args.out_dir}/models.pt')


if __name__ == "__main__":
    main(parse_arguments())
