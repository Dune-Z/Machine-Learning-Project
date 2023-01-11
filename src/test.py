import argparse
import numpy as np
import pandas as pd
from utils import evaluate_model


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",
                        type=str,
                        default="../models",
                        help="Directory to save the model.")
    parser.add_argument("--data",
                        type=str,
                        default="../data/test_data_sample.json",
                        help="Path to the testing data.")
    return parser.parse_args()


def main(args):
    from torch import load
    prep = load(f'{args.in_dir}/preprocessor.pt')
    models = load(f'{args.in_dir}/models.pt')

    # Load data from disk
    test_df = pd.read_json(args.data)

    # Cleanse and transform data
    test_df = prep.cleanse(test_df)
    test_df_prep = prep.transform(test_df)

    # Get feature matrix and target vector
    X = test_df_prep.drop('fit', axis=1).values
    y = test_df_prep['fit'].values

    # To tackle class imbalance, we split the majority class (True to Size) into 3 folds,
    # and train the model on each fold separately. When predicting on the test set,
    # we aggregate the predictions from all 3 trained models and take the majority vote.
    y_preds = [model.predict(X) for model in models]
    y_preds = list(map(list, zip(*y_preds)))  # list transpose
    y_pred = np.array([max(set(votes), key=votes.count) for votes in y_preds])
    print(evaluate_model(y, y_pred))


if __name__ == "__main__":
    main(parse_arguments())
