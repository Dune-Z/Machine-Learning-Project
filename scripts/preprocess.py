import json
import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self, path: str):
        with open(path, 'r') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)

    def preprocess(self, is_train=False):
        """
        Preprocess the data, Yunqin Zhu's version.
        """
        self._arrange_cols(is_train)
        self._drop_ufeff()
        self._empty_to_nan()
        self._height_to_cm()
        self._weight_to_kg()
        self._bust_size_split()
        self._body_type_to_cat()
        self._preprocess_age()
        self._preprocess_size()
        self._preprocess_price()
        self._preprocess_rent_for()
        self._preprocess_item_name()
        self._preprocess_usually_wear()

    def _arrange_cols(self, is_train=False):
        """
        Rearrange columns into the order in the data description.
        """
        if is_train:
            self.df = self.df[[
                'fit', 'item_name', 'size', 'price', 'user_name', 'rented_for',
                'usually_wear', 'age', 'height', 'weight', 'body_type',
                'bust_size', 'review_summary', 'review', 'rating'
            ]].copy()
        else:
            self.df = self.df[[
                'fit', 'item_name', 'size', 'price', 'rented_for', 'usually_wear',
                'age', 'height', 'weight', 'body_type', 'bust_size'
            ]].copy()

    def _drop_ufeff(self):
        """
        Drop the rows corrupted by the byte-order mark '\udeff' in the data.
        """
        self.df.drop(self.df[self.df.item_name.str.contains('\ufeff')].index, inplace=True)

    def _empty_to_nan(self):
        """
        Convert empty strings to NaN.
        """
        self.df.replace('', np.nan, inplace=True)

    def _height_to_cm(self):
        """
        Convert height from feet and inches to centimeters.
        Set invalid values to NaN.
        """
        self.df.height = self.df.height.astype(str, copy=False)
        pos = self.df.height.str.match(r'^\d+\' \d+\"$')
        self.df.loc[~pos, 'height'] = np.nan
        # Convert feet and inches to centimeters
        temp = self.df.height[pos].str.extract(r'(\d+)\' (\d+)\"').astype(int, copy=False)
        self.df.loc[pos, 'height'] = (temp[0] * 12 + temp[1]) * 2.54
        self.df.height = self.df.height.astype(float, copy=False)
        # Set outliers (>200 cm) to NaN
        self.df.loc[self.df.height > 200, 'height'] = np.nan

    def _weight_to_kg(self):
        """
        Convert weight from pounds to kilograms.
        Set invalid values to NaN.
        """
        self.df.weight = self.df.weight.astype(str, copy=False)
        pos = self.df.weight.str.match(r'^\d+LBS$')
        self.df.loc[~pos, 'weight'] = np.nan
        # Convert pounds to kilograms
        self.df.loc[pos, 'weight'] = self.df.weight[pos].str.extract(r'^(\d+)LBS$',
                                                           expand=False)
        self.df.weight = self.df.weight.astype(float, copy=False)
        self.df.weight = self.df.weight * 0.45359237
        # Set outliers (<30 kg or >150 kg) to NaN
        self.df.loc[self.df.weight < 30, 'weight'] = np.nan
        self.df.loc[self.df.weight > 150, 'weight'] = np.nan

    def _bust_size_split(self):
        """
        Split bust_size into 2 features:
            - bust_size: number part in inches, as float
            - cup_size: letter part, as ordinal category
        """
        self.df.bust_size = self.df.bust_size.astype(str, copy=False)
        pos = self.df.bust_size.str.match(r'^\d+[A-K].*$')
        self.df.loc[~pos, 'bust_size'] = np.nan
        # Split the number and the letter
        temp = self.df.bust_size[pos].str.extract(r'^(\d+)([A-K].*)$')
        self.df.loc[pos, 'bust_size'] = temp[0]
        self.df.bust_size = self.df.bust_size.astype(float, copy=False)
        self.df['cup_size'] = np.nan
        self.df.loc[pos, 'cup_size'] = temp[1]
        self.df.cup_size = self.df.cup_size.astype('category', copy=False)
        self.df.cup_size = self.df.cup_size.cat.set_categories(
            'AA A B C D D+ DD DDD/E F G H I J'.split(), ordered=True)

    def _body_type_to_cat(self):
        """
        Convert body_type to category.
        """
        self.df.body_type = self.df.body_type.astype('category', copy=False)

    def _preprocess_age(self):
        """
        Data preprocess: age column
        1. Convert all non-pure number item into NaN
        2. Convert value >= 100 item into NaN\n
        After process:\n
        class: 84, dtype: int64
        """
        self.df['age'] = self.df.age.astype(str, copy=False)
        # Convert all non-pure number item into NaN
        self.df['age'][~self.df.age.str.match(r'^[0-9]*$')] = np.nan
        # Convert value >= 100 item into NaN
        self.df['age'][self.df['age'].astype(float) >= 100] = np.nan

    def _preprocess_usually_wear(self):
        """
        Data preprocess: age column\n
        Convert all non-pure number item into NaN
        After process:\n
        class: 50, dtype: int64
        """
        self.df['usually_wear'] = self.df.usually_wear.astype(str, copy=False)
        # Convert all non-pure number item into NaN
        self.df['usually_wear'][~self.df.usually_wear.str.match(r'^[0-9]*$')] = np.nan

    def _preprocess_item_name(self):
        """
        Split item name by '\n', which fairly reduce the category number.
        """
        splitting = lambda x: x.split('\n') if '\n' in x else list(x) + [np.nan]
        item_names = [splitting(content) for content in self.df.item_name]
        item_names = [content + ['None'] if len(content) == 1 else content for content in item_names]
        item_names = list(map(list, zip(*item_names)))
        self.df.insert(0, 'item_name1', np.array(item_names[0]))
        self.df.insert(1, 'item_name2', np.array(item_names[1]))
        self.df.drop('item_names', axis=1, inplace=True)

    def _preprocess_rent_for(self):
        """
        Already clean, convert column value type to category.
        """
        self.df['rent_for'] = self.df.rented_for.astype('category', copy=False)

    def _preprocess_price(self):
        """
        remove '$' and convert column value in float.
        """
        self.df['price'] = self.df.price.apply(lambda x: float(x[1:]))

    def _preprocess_size(self):
        # TODO: preprocess 'size' feature.
        pass
