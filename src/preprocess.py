import numpy as np
import pandas as pd
import warnings


class Preprocessor:

    def __init__(self, pipeline=[]):
        self.train_df = None
        self.test_df = None
        self.pipeline = pipeline

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all transformers and transform the training data.
        """
        self.train_df = df
        for i, dt in enumerate(self.pipeline):
            print(type(dt))
            if dt.cols is not None:
                df = dt.fit_transform(df)
                continue
            # SelectOutputColumns
            for target_dt in self.pipeline:
                if dt.target == target_dt.name:
                    self.pipeline[i + 1].cols += (target_dt.out_cols)
                    break

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the test data.
        """
        self.test_df = df
        for dt in self.pipeline:
            if dt.cols is not None:
                df = dt.transform(df)
        return df

    def cleanse(self,
                df: pd.DataFrame,
                is_train=False,
                exclude_cols=[]) -> pd.DataFrame:
        """
        Cleanse the training or test data.
        """
        if is_train:
            self.train_df = df
        else:
            self.test_df = df

        df = self.cleanse_before(df, is_train=is_train)
        # label
        if 'fit' not in exclude_cols:
            df = self.cleanse_fit(df)
        # item attributes
        if 'item_name' not in exclude_cols:
            df = self.cleanse_item_name(df)
        if 'size' not in exclude_cols:
            df = self.cleanse_size(df)
        if 'price' not in exclude_cols:
            df = self.cleanse_price(df)
        # transaction info
        if 'rented_for' not in exclude_cols:
            df = self.cleanse_rented_for(df)
        if 'usually_wear' not in exclude_cols:
            df = self.cleanse_usually_wear(df)
        # user attributes
        if is_train and 'body_type' not in exclude_cols:
            df = self.cleanse_user_name(df)
        if 'age' not in exclude_cols:
            df = self.cleanse_age(df)
        if 'height' not in exclude_cols:
            df = self.cleanse_height(df)
        if 'weight' not in exclude_cols:
            df = self.cleanse_weight(df)
        if 'body_type' not in exclude_cols:
            df = self.cleanse_body_type(df)
        if 'bust_size' not in exclude_cols:
            df = self.cleanse_bust_size(df)
        # feedback info
        if is_train:
            if 'review_summary' not in exclude_cols:
                df = self.cleanse_review_summary(df)
            if 'review' not in exclude_cols:
                df = self.cleanse_review(df)
            if 'rating' not in exclude_cols:
                df = self.cleanse_rating(df)
        if not exclude_cols:
            df = self.cleanse_after(df, is_train=is_train)
        return df

    @staticmethod
    def cleanse_before(df: pd.DataFrame, is_train=False):
        """
        This function is called before cleansing each column.
        - Change column order for consistency.
        - Replace empty strings with NaN.
        - Drop rows corrupted by the byte-order mark '\\udeff'.
        - Replace 'fit' strings with numbers.
        """
        # Drop rows corrupted by the byte-order mark '\\udeff'.
        pos = df['item_name'].str.contains('\ufeff', na=False)
        if is_train:
            df.drop(df[pos].index, inplace=True)
        else:
            df.loc[pos, :] = np.nan

        # Replace empty strings with NaN.
        df.replace('', np.nan, inplace=True)

        # Drop rows without 'user_name'
        if is_train:
            df.dropna(subset=['user_name'], inplace=True)

        # Replace 'fit' strings with numbers.
        if is_train:
            df['fit'].replace({
                'Small': '1',
                'True to Size': '2',
                'Large': '3'
            },
                              inplace=True)
        return df

    @staticmethod
    def cleanse_fit(df: pd.DataFrame):
        """
        Cleanse the label 'fit'.
        - Set value type as ordered category.
        """
        df['fit'] = df['fit'].astype('category', copy=False)
        df['fit'] = df['fit'].cat.set_categories(['1', '2', '3'], ordered=True)
        return df

    @staticmethod
    def cleanse_item_name(df: pd.DataFrame):
        """
        Cleanse the feature 'item_name'.
        - Split strings by '\n', yielding two columns: 'brand' and 'item_name'.
        - Set 'brand' as NaN for items without brand.
        - Extract 'category' from 'item_name'.
        """
        # Split strings by '\n'.
        new_cols = df['item_name'].str.split('\n', expand=True)
        df['brand'] = new_cols[0]
        df['item_name'] = new_cols[0] + '_' + new_cols[1]
        # Set 'brand' as NaN for items without brand.
        pos = df['brand'].str.endswith('"', na=False)
        df.loc[pos, 'item_name'] = df.loc[pos, 'brand'].str.removesuffix('"')
        df.loc[pos, 'brand'] = np.nan
        # Extract 'category' from 'item_name'.
        df['category'] = df['item_name'].str.extract(r'\b(\w+)$')
        return df

    @staticmethod
    def cleanse_size(df: pd.DataFrame):
        """
        Cleanse the feature 'size'. \n
        ! This function should be called after 'item_name' is cleansed.
        - Set 'None', 'NONE', '-1' as NaN.
        - Get 'size_scheme', 'size_main' and 'size_suffix' of each transaction.
        - Fix 'size_scheme' for each item.
        """
        # Set 'None', 'NONE', '-1' as NaN.
        pos = df['size'].str.match(r'^None|NONE|-1$', na=False)
        df.loc[pos, 'size'] = np.nan
        df['size'] = df['size'].astype(str, copy=False)

        # Get 'size_scheme', 'size_main' and 'size_suffix' of each transaction.
        size_strings = pd.DataFrame()
        size_strings[['item_name', 'size']] = df[['item_name',
                                                  'size']].astype(str)
        size_strings['size_scheme'] = np.nan

        # - Case 1: size in letter
        size_strings = size_strings.join(size_strings['size'].str.extract(
            r'^(XXS|\d*XS|S|M|L|\d*XL|XXL|\d*X)([PRL])?$'))
        size_strings.loc[size_strings[0].notna(), 'size_scheme'] = 'letter'

        # - Case 2: size in number
        temp = size_strings['size'].str.extract(r'^(\d+)(W|W?[PRL])?$')
        pos = temp[0].notna()
        size_strings.loc[pos, [0, 1]] = temp.loc[pos, :]
        size_strings.loc[pos, 'size_scheme'] = 'number'

        # - Case 3: size between 2 letter
        temp = size_strings['size'].str.extract(
            r'^((?:XXS|XS|P|S|M|L|XL|XXL|X)(?:-|\/)(?:XXS|XS|P|S|M|L|XL|XXL|X))([PRL])?$'
        )
        temp[0] = temp[0].str.replace(r'\/', '-', regex=True)
        pos = temp[0].notna()
        size_strings.loc[pos, [0, 1]] = temp.loc[pos, :]
        size_strings.loc[pos, 'size_scheme'] = 'letter'

        # - Case 4: size between 2 number
        temp = size_strings['size'].str.extract(
            r'^((?:\d+)(?:-|\/)(?:\d+))([PRL])?$')
        temp[0] = temp[0].str.replace(r'\/', '-', regex=True)
        pos = temp[0].notna()
        size_strings.loc[pos, [0, 1]] = temp.loc[pos, :]
        size_strings.loc[pos, 'size_scheme'] = 'number'

        size_strings.loc[size_strings['size'] == 'ONESIZE',
                         'size_scheme'] = 'onesize'
        size_strings['size'].replace({'ONESIZE': 'unknown'}, inplace=True)
        size_strings['size'].fillna('unknown', inplace=True)

        # Fix 'size_scheme' for each item.
        item_indices = df.groupby(['item_name']).indices
        item_size_schemes = size_strings.groupby(['item_name'
                                                  ])['size_scheme'].unique()
        for name, schemes in item_size_schemes.items():
            if len(schemes) == 1:
                pass
            elif 'onesize' in schemes:
                size_strings.iloc[item_indices[name], 1] = 'unknown'
                size_strings.iloc[item_indices[name], 2] = 'onesize'
                size_strings.iloc[item_indices[name], 3] = np.nan
            elif 'letter' in schemes and 'number' in schemes:
                size_strings.iloc[item_indices[name], 2] = 'mixed'
            elif 'number' in schemes:
                size_strings.iloc[item_indices[name], 2] = 'number'
            elif 'letter' in schemes:
                size_strings.iloc[item_indices[name], 2] = 'letter'

        size_strings.rename(columns={
            0: 'size_main',
            1: 'size_suffix'
        },
                            inplace=True)

        df = df.join(size_strings[['size_main', 'size_suffix', 'size_scheme']])

        return df

    @staticmethod
    def cleanse_price(df: pd.DataFrame):
        """
        Cleanse the feature 'price'.
        - Set invalid values as NaN.
        - Remove the dollar sign '$'.
        """
        # Set invalid values as NaN.
        pos = df['price'].str.match(r'^\$\d+$', na=False)
        df.loc[~pos, 'price'] = np.nan
        # Remove the dollar sign '$'.
        df['price'] = df['price'].str.removeprefix('$')
        df['price'] = df['price'].astype(float, copy=False)
        return df

    @staticmethod
    def cleanse_rented_for(df: pd.DataFrame):
        """
        Cleanse the feature 'rented_for'.
        """
        return df

    @staticmethod
    def cleanse_usually_wear(df: pd.DataFrame):
        """
        Cleanse the feature 'usually_wear'.
        - Set invalid values as NaN.
        """
        df['usually_wear'] = df['usually_wear'].astype('string', copy=False)
        pos = df['usually_wear'].str.match(r'^\d+$', na=False)
        df.loc[~pos, 'usually_wear'] = np.nan
        df['usually_wear'] = df['usually_wear'].astype(float, copy=False)
        return df

    @staticmethod
    def cleanse_user_name(df: pd.DataFrame):
        """
        Cleanse the feature 'user_name'.
        - Fill NaN with 'RTR Customer'.
        """
        df['user_name'].fillna('RTR Customer', inplace=True)
        return df

    @staticmethod
    def cleanse_age(df: pd.DataFrame):
        """
        Cleanse the feature 'age'.
        - Set invalid values as NaN.
        - Set outliers (<=5 or >=100) as NaN.
        """
        # Set invalid values as NaN.
        pos = df['age'].str.match(r'^\d+$', na=False)
        df.loc[~pos, 'age'] = np.nan
        df['age'] = df['age'].astype(float, copy=False)
        # Set outliers (<=5 or >=100) to NaN.
        df.loc[df['age'] <= 5, 'age'] = np.nan
        df.loc[df['age'] >= 100, 'age'] = np.nan
        return df

    @staticmethod
    def cleanse_height(df: pd.DataFrame):
        """
        Cleanse the feature 'height'.
        - Set invalid values as NaN.
        - Convert height values from feet and inches to centimeters.
        - Set outliers (>200 cm) as NaN.
        """
        # Set invalid values as NaN.
        pos = df['height'].str.match(r'^\d+\' \d+\"$', na=False)
        df.loc[~pos, 'height'] = np.nan
        # Convert height values from feet and inches to centimeters.
        temp = df.loc[pos, 'height'].str.extract(r'(\d+)\' (\d+)\"').astype(
            int, copy=False)
        df['height'] = np.nan
        df.loc[pos, 'height'] = (temp[0] * 12 + temp[1]) * 2.54
        # Set outliers (>200 cm) as NaN.
        df.loc[df['height'] > 200, 'height'] = np.nan
        return df

    @staticmethod
    def cleanse_weight(df: pd.DataFrame):
        """
        Cleanse the feature 'weight'.
        - Set invalid values as NaN.
        - Convert weight values from pounds to kilograms.
        - Set outliers (<30 kg or >150 kg) as NaN
        """
        # Set invalid values as NaN.
        pos = df['weight'].str.match(r'^\d+LBS$', na=False)
        df.loc[~pos, 'weight'] = np.nan
        # Convert weight values from pounds to kilograms.
        df.loc[pos, 'weight'] = df.loc[pos,
                                       'weight'].str.extract(r'^(\d+)LBS$',
                                                             expand=False)
        df['weight'] = df['weight'].astype(float, copy=False)
        df['weight'] = df['weight'] * 0.45359237
        # Set outliers (<30 kg or >150 kg) as NaN.
        df.loc[df['weight'] < 30, 'weight'] = np.nan
        df.loc[df['weight'] > 150, 'weight'] = np.nan
        return df

    @staticmethod
    def cleanse_body_type(df: pd.DataFrame):
        """
        Cleanse the feature 'body_type'.
        """
        return df

    @staticmethod
    def cleanse_bust_size(df: pd.DataFrame):
        """
        Cleanse the feature 'bust_size'.
        - Set invalid values as NaN.
        - Split 'bust_size' into 2 features:
            - 'bust_size': number part in inches, as float
            - 'cup_size': letter part, as ordered category
        """
        # Set invalid values as NaN.
        pos = df['bust_size'].str.match(r'^\d+[A-K].*$', na=False)
        df.loc[~pos, 'bust_size'] = np.nan
        # Split 'bust_size' into 2 features.
        temp = df.loc[pos, 'bust_size'].str.extract(r'^(\d+)([A-K].*)$')
        df.loc[pos, 'bust_size'] = temp[0]
        df['bust_size'] = df['bust_size'].astype(float, copy=False)
        df['cup_size'] = np.nan
        df.loc[pos, 'cup_size'] = temp[1]
        df['cup_size'] = df['cup_size'].astype('category', copy=False)
        df['cup_size'] = df['cup_size'].cat.set_categories(
            'AA A B C D D+ DD DDD/E F G H I J'.split(), ordered=True)
        return df

    @staticmethod
    def cleanse_review_summary(df: pd.DataFrame):
        """
        Cleanse the feature 'review_summary'.
        """
        return df

    @staticmethod
    def cleanse_review(df: pd.DataFrame):
        """
        Cleanse the feature 'review'.
        """
        return df

    @staticmethod
    def cleanse_rating(df: pd.DataFrame):
        """
        Cleanse the feature 'rating'.
        - Set invalid values as NaN.
        """
        # Set invalid values as NaN.
        pos = df['rating'].str.match(r'^\d+$', na=False)
        df.loc[~pos, 'rating'] = np.nan
        df['rating'] = df['rating'].astype(float, copy=False)
        df.loc[df['rating'] < 1, 'rating'] = np.nan
        df.loc[df['rating'] > 5, 'rating'] = np.nan
        return df

    @staticmethod
    def cleanse_after(df: pd.DataFrame, is_train=False):
        """
        This function is called after cleansing each column.
        - Change column order for consistency.
        """
        if is_train:
            df = df.reindex(columns=[
                'fit', 'item_name', 'brand', 'category', 'size', 'size_main',
                'size_suffix', 'size_scheme', 'price', 'user_name',
                'rented_for', 'usually_wear', 'age', 'height', 'weight',
                'body_type', 'bust_size', 'cup_size', 'review_summary',
                'review', 'rating'
            ],
                            copy=False)
        else:
            df = df.reindex(columns=[
                'fit', 'item_name', 'brand', 'category', 'size', 'size_main',
                'size_suffix', 'size_scheme', 'price', 'rented_for',
                'usually_wear', 'age', 'height', 'weight', 'body_type',
                'bust_size', 'cup_size'
            ],
                            copy=False)
        return df


class DataTransformer:
    """
    The base class for data transformers.
    """

    def __init__(self, cols=[], name=''):
        self.name = name
        self.cols = cols
        self.out_cols = []  # output columns


class OrdinalEncoder(DataTransformer):
    """
    Encode ordered categorical features as integers in the same columns.
    If column is pd.Categorical with order, then its codes are returned.\n
    Otherwise, order can be passed as a dict of column name and category list.\n
    If neither, then categories are sorted by frequency in ascending order.\n
    NaN is encoded as np.nan.
    """

    def __init__(self, cols=[], order={}, name=''):
        super().__init__(cols, name)
        self.order = order

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the encoded data
        """
        self.out_cols = self.cols
        for col in self.cols:
            df[col] = df[col].astype('category', copy=False)
            if not df[col].cat.ordered:
                if col in self.order:
                    df[col] = df[col].cat.set_categories(self.order[col],
                                                         ordered=True)
                else:
                    df[col] = df[col].cat.set_categories(
                        df[col].value_counts().keys(), ordered=True)
            self.order[col] = df[col].cat.categories.tolist()
            df[col] = df[col].cat.codes
            df[col].replace(-1, np.nan, inplace=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the encoded test data
        """
        for col in self.cols:
            df[col] = df[col].astype('category', copy=False)
            df[col] = df[col].cat.set_categories(self.order[col], ordered=True)
            df[col] = df[col].cat.codes
            df[col].replace(-1, np.nan, inplace=True)
        return df


class OneHotEncoder(DataTransformer):
    """
    Encode categorical features as a one-hot vector, with each category as a new column.\n
    NaN and unseen categories are encoded as zeros in each column.\n
    This encoder is extremely slow for high cardinality features (since I use for loop). Do not use it in such cases.
    """

    def __init__(self, cols=[], max_categories=500, name=''):
        super().__init__(cols, name)
        self.max_categories = max_categories
        self.categories = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the encoded data
        """
        dummies = pd.get_dummies(df[self.cols])
        self.out_cols = dummies.columns.tolist()
        for col in self.cols:
            self.categories[col] = df[col].dropna().unique().tolist()
            if len(self.categories[col]) > self.max_categories:
                warnings.warn(
                    f"Column {col} has {len(self.categories[col])} categories, which is more than the maximum of {self.max_categories}. Please consider using other encoders."
                )
        df = df.join(dummies)
        df.drop(self.cols, axis=1, inplace=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the encoded test data
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in self.cols:
                for category in self.categories[col]:
                    df[f'{col}_{category}'] = (df[col] == category).astype(int)
        df.drop(self.cols, axis=1, inplace=True)
        return df
    
    
class TargetEncoder:
    """
    Replacing categories by the mean value of target of category variables\n
    Target should be numeric 
    """
    def __init__(self, cols: list = [], target: str = 'weight'):
        self.cols = cols
        self.target = target
        self.mapping = {}
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the encoded data
        """
        for cols in self.cols:
            mapping = df.groupby(cols).mean(numeric_only=True)[self.target].to_dict()
            self.mapping[cols] = mapping
            df[cols].replace(mapping, inplace=True)
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the encoded test data
        """
        for cols in self.cols:
            mapping = self.mapping[cols]
            df[cols] = df[cols].map(mapping) 
        return df
            

class StandardScaler(DataTransformer):
    """
    Standardize features by removing the mean and scaling to unit variance.
    """

    def __init__(self, cols=[], name=''):
        super().__init__(cols, name)
        self.mean = {}
        self.std = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the scaled data
        """
        self.out_cols = self.cols
        for col in self.cols:
            self.mean[col] = df[col].mean()
            self.std[col] = df[col].std()
            df[col] = (df[col] - self.mean[col]) / self.std[col]
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the scaled test data
        """
        for col in self.cols:
            df[col] = (df[col] - self.mean[col]) / self.std[col]
        return df


class MinMaxScaler(DataTransformer):
    """
    Standardize features by removing the mean and scaling to [0, 1].
    """

    def __init__(self, cols=[], name=''):
        super().__init__(cols, name)
        self.min = {}
        self.max = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the scaled data
        """
        self.out_cols = self.cols
        for col in self.cols:
            self.min[col] = df[col].min()
            self.max[col] = df[col].max()
            df[col] = (df[col] - self.min[col]) / (self.max[col] -
                                                   self.min[col])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the scaled test data
        """
        for col in self.cols:
            df[col] = (df[col] - self.min[col]) / (self.max[col] -
                                                   self.min[col])
        return df


class MeanImputer(DataTransformer):
    """
    Impute missing values with the mean along each column.
    """

    def __init__(self, cols=[], name=''):
        super().__init__(cols, name)
        self.mean = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the imputed data
        """
        self.out_cols = self.cols
        for col in self.cols:
            self.mean[col] = df[col].mean()
            df[col].fillna(self.mean[col], inplace=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the imputed test data
        """
        for col in self.cols:
            df[col].fillna(self.mean[col], inplace=True)
        return df


class MedianImputer(DataTransformer):
    """
    Impute missing values with the median along each column.
    """

    def __init__(self, cols=[], name=''):
        super().__init__(cols, name)
        self.median = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the imputed data
        """
        self.out_cols = self.cols
        for col in self.cols:
            self.median[col] = df[col].median()
            df[col].fillna(self.median[col], inplace=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the imputed test data
        """
        for col in self.cols:
            df[col].fillna(self.median[col], inplace=True)
        return df


class ModeImputer(DataTransformer):
    """
    Impute missing values with the mode along each column.
    """

    def __init__(self, cols=[], name=''):
        super().__init__(cols, name)
        self.mode = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the imputed data
        """
        self.out_cols = self.cols
        for col in self.cols:
            self.mode[col] = df[col].mode()[0]
            df[col].fillna(self.mode[col], inplace=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the imputed test data
        """
        for col in self.cols:
            df[col].fillna(self.mode[col], inplace=True)
        return df


class ConstantImputer(DataTransformer):
    """
    Impute missing values with a constant value.
    """

    def __init__(self, cols=[], value=0, name=''):
        super().__init__(cols, name)
        self.value = value

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the training data and return the imputed data
        """
        self.out_cols = self.cols
        for col in self.cols:
            df[col].fillna(self.value, inplace=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the imputed test data
        """
        for col in self.cols:
            df[col].fillna(self.value, inplace=True)
        return df


class DropColumns(DataTransformer):
    """
    Drop columns from a dataframe.
    """

    def __init__(self, cols=[], name=''):
        super().__init__(cols, name)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        return df


class SelectOutputColumns:
    """
    Select the output columns of a named DataTransformer.\n
    The main logic is implemented in the Preprocessor class as I have not found a better way to do this.
    """

    def __init__(self, target: str):
        self.cols = None
        self.target = target
