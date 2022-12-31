import pandas as pd
import numpy as np


class Preprocessor:

    train_df: pd.DataFrame = None
    test_df: pd.DataFrame = None

    def __init__(self):
        pass

    def fit_transform(self, train_df: pd.DataFrame):
        """
        Fit the model using the training data.
        """
        self.train_df = train_df
        # Do something

    def transform(self, test_df: pd.DataFrame):
        """
        Transform the testing data.
        """
        self.test_df = test_df
        test_df = proc_before(test_df, is_train=False)
        test_df = proc_fit(test_df)
        test_df = proc_item_name(test_df)
        test_df = proc_size(test_df)
        test_df = proc_price(test_df)
        test_df = proc_rented_for(test_df)
        test_df = proc_usually_wear(test_df)
        test_df = proc_age(test_df)
        test_df = proc_height(test_df)
        test_df = proc_weight(test_df)
        test_df = proc_body_type(test_df)
        test_df = proc_bust_size(test_df)
        test_df = proc_after(test_df)
        return test_df


def proc_before(df: pd.DataFrame, is_train=False):
    """
    This function is called before processing each column.
    - Change column order for consistency.
    - Drop rows corrupted by the byte-order mark '\\udeff'.
    - Replace empty strings with NaN.
    """
    # Change column order for consistency.
    if is_train:
        df = df[[
            'fit', 'item_name', 'size', 'price', 'user_name', 'rented_for',
            'usually_wear', 'age', 'height', 'weight', 'body_type',
            'bust_size', 'review_summary', 'review', 'rating'
        ]].copy()
    else:
        df = df[[
            'fit', 'item_name', 'size', 'price', 'rented_for', 'usually_wear',
            'age', 'height', 'weight', 'body_type', 'bust_size'
        ]].copy()
    # Drop rows corrupted by the byte-order mark '\\udeff'.
    df.drop(df[df.item_name.str.contains('\ufeff')].index, inplace=True)
    # Replace empty strings with NaN.
    df.replace('', np.nan, inplace=True)
    return df


def proc_fit(df: pd.DataFrame):
    """
    Preprocess the label 'fit'.
    - Set value type as category.
    """
    df.fit = df.fit.astype('category', copy=False)
    return df


def proc_item_name(df: pd.DataFrame):
    """
    Preprocess the feature 'item_name'.
    - Split strings by '\n', yielding two columns: 'item_name1' and 'item_name2'.
    """
    # Split strings by '\n'.
    new_cols = df.item_name.str.split('\n', expand=True)
    df.insert(1, 'item_name1', new_cols[0])
    df.insert(2, 'item_name2', new_cols[1])
    df.drop(columns=['item_name'], inplace=True)
    #
    pos = df.item_name1.str.endswith('"')
    df.loc[pos, 'item_name2'] = df.item_name1[pos].str.removesuffix('"')
    df.loc[pos, 'item_name1'] = np.nan
    df.item_name1 = df.item_name1.astype('category', copy=False)
    df.item_name2 = df.item_name2.astype('category', copy=False)
    return df


def proc_size(df: pd.DataFrame):
    """
    Preprocess the feature 'size'.
    - Set 'None', 'NONE', '-1' as NaN
    """
    df['size'] = df['size'].astype('string', copy=False)
    pos = df['size'].str.match(r'^None|NONE|-1$')
    df.loc[pos, 'size'] = np.nan
    return df


def proc_price(df: pd.DataFrame):
    """
    Preprocess the feature 'price'.
    - Remove the dollar sign '$'.
    """
    df.price = df.price.astype('string', copy=False)
    df.price = df.price.str.removeprefix('$')
    df.price = df.price.astype(float, copy=False)
    return df


def proc_rented_for(df: pd.DataFrame):
    """
    Preprocess the feature 'rented_for'.
    - Set value type as category.
    """
    df.rented_for = df.rented_for.astype('category', copy=False)
    return df


def proc_usually_wear(df: pd.DataFrame):
    """
    Preprocess the feature 'usually_wear'.
    - Set invalid values as NaN.
    """
    df.usually_wear = df.usually_wear.astype('string', copy=False)
    pos = df.usually_wear.str.match(r'^[0-9]*$')
    df.loc[~pos, 'usually_wear'] = np.nan
    df.usually_wear = df.usually_wear.astype(float, copy=False)
    return df


def proc_age(df: pd.DataFrame):
    """
    Preprocess the feature 'age'.
    - Set invalid values as NaN.
    - Set outliers (>=100) as NaN.
    """
    # Set invalid values as NaN.
    df.age = df.age.astype('string', copy=False)
    pos = df.age.str.match(r'^[0-9]*$')
    df.loc[~pos, 'age'] = np.nan
    df.age = df.age.astype(float, copy=False)
    # Set outliers (>=100) to NaN.
    df.loc[df.age >= 100, 'age'] = np.nan
    return df


def proc_height(df: pd.DataFrame):
    """
    Preprocess the feature 'height'.
    - Set invalid values as NaN.
    - Convert height values from feet and inches to centimeters.
    - Set outliers (>200 cm) as NaN.
    """
    # Set invalid values as NaN.
    df.height = df.height.astype('string', copy=False)
    pos = df.height.str.match(r'^\d+\' \d+\"$')
    df.loc[~pos, 'height'] = np.nan
    # Convert height values from feet and inches to centimeters.
    temp = df.height[pos].str.extract(r'(\d+)\' (\d+)\"').astype(int,
                                                                 copy=False)
    df.height = np.nan
    df.loc[pos, 'height'] = (temp[0] * 12 + temp[1]) * 2.54
    # Set outliers (>200 cm) as NaN.
    df.loc[df.height > 200, 'height'] = np.nan
    return df


def proc_weight(df: pd.DataFrame):
    """
    Preprocess the feature 'weight'.
    - Set invalid values as NaN.
    - Convert weight values from pounds to kilograms.
    - Set outliers (<30 kg or >150 kg) as NaN
    """
    # Set invalid values as NaN.
    df.weight = df.weight.astype('string', copy=False)
    pos = df.weight.str.match(r'^\d+LBS$')
    df.loc[~pos, 'weight'] = np.nan
    # Convert weight values from pounds to kilograms.
    df.loc[pos, 'weight'] = df.weight[pos].str.extract(r'^(\d+)LBS$',
                                                       expand=False)
    df.weight = df.weight.astype(float, copy=False)
    df.weight = df.weight * 0.45359237
    # Set outliers (<30 kg or >150 kg) as NaN.
    df.loc[df.weight < 30, 'weight'] = np.nan
    df.loc[df.weight > 150, 'weight'] = np.nan
    return df


def proc_body_type(df: pd.DataFrame):
    """
    Preprocess the feature 'body_type'.
    - Set value type as category.
    """
    df.body_type = df.body_type.astype('category', copy=False)
    return df


def proc_bust_size(df: pd.DataFrame):
    """
    Preprocess the feature 'bust_size'.
    - Set invalid values as NaN.
    - Split 'bust_size' into 2 features:
        - 'bust_size': number part in inches, as float
        - 'cup_size': letter part, as ordinal category
    """
    df.bust_size = df.bust_size.astype('string', copy=False)
    pos = df.bust_size.str.match(r'^\d+[A-K].*$')
    df.loc[~pos, 'bust_size'] = np.nan
    # Split 'bust_size' into 2 features.
    temp = df.bust_size[pos].str.extract(r'^(\d+)([A-K].*)$')
    df.loc[pos, 'bust_size'] = temp[0]
    df.bust_size = df.bust_size.astype(float, copy=False)
    df['cup_size'] = np.nan
    df.loc[pos, 'cup_size'] = temp[1]
    df.cup_size = df.cup_size.astype('category', copy=False)
    df.cup_size = df.cup_size.cat.set_categories(
        'AA A B C D D+ DD DDD/E F G H I J'.split(), ordered=True)
    return df


def proc_after(df: pd.DataFrame):
    """
    This function is called after processing each column.
    """
    return df