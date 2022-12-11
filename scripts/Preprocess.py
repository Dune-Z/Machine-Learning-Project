# All preprocess functions are here

def arrange_cols(df: pd.DataFrame, is_train=False):
    '''
    Rearrange columns into the order in the data description.
    '''
    if is_train:
        df = df[[
            'fit', 'item_name', 'size', 'price', 'user_name', 'rented_for',
            'usually_wear', 'age', 'height', 'weight', 'body_type',
            'bust_size', 'review_summary', 'review', 'rating'
        ]].copy()
        return df
    else:
        df = df[[
            'fit', 'item_name', 'size', 'price', 'rented_for', 'usually_wear',
            'age', 'height', 'weight', 'body_type', 'bust_size'
        ]].copy()
        return df


def drop_ufeff(df: pd.DataFrame):
    '''
    Drop the rows corrupted by the byte-order mark '\udeff' in the data.
    '''
    df.drop(df[df.item_name.str.contains('\ufeff')].index, inplace=True)
    return df


def empty_to_nan(df: pd.DataFrame):
    '''
    Convert empty strings to NaN.
    '''
    df.replace('', np.nan, inplace=True)
    return df
  
def height_to_cm(df: pd.DataFrame):
    '''
    Convert height from feet and inches to centimeters.
    Set invalid values to NaN.
    '''
    df.height = df.height.astype(str, copy=False)
    pos = df.height.str.match(r'^\d+\' \d+\"$')
    df.loc[~pos, 'height'] = np.nan
    # Convert feet and inches to centimeters
    temp = df.height[pos].str.extract(r'(\d+)\' (\d+)\"').astype(int,
                                                                 copy=False)
    df.loc[pos, 'height'] = (temp[0] * 12 + temp[1]) * 2.54
    df.height = df.height.astype(float, copy=False)
    # Set outliers (>200 cm) to NaN
    df.loc[df.height > 200, 'height'] = np.nan
    return df


def weight_to_kg(df: pd.DataFrame):
    '''
    Convert weight from pounds to kilograms.
    Set invalid values to NaN.
    '''
    df.weight = df.weight.astype(str, copy=False)
    pos = df.weight.str.match(r'^\d+LBS$')
    df.loc[~pos, 'weight'] = np.nan
    # Convert pounds to kilograms
    df.loc[pos, 'weight'] = df.weight[pos].str.extract(r'^(\d+)LBS$',
                                                       expand=False)
    df.weight = df.weight.astype(float, copy=False)
    df.weight = df.weight * 0.45359237
    # Set outliers (<30 kg or >150 kg) to NaN
    df.loc[df.weight < 30, 'weight'] = np.nan
    df.loc[df.weight > 150, 'weight'] = np.nan
    return df


def bust_size_split(df: pd.DataFrame):
    '''
    Split bust_size into 2 features:
        - bust_size: number part in inches, as float
        - cup_size: letter part, as ordinal category
    '''
    df.bust_size = df.bust_size.astype(str, copy=False)
    pos = df.bust_size.str.match(r'^\d+[A-K].*$')
    df.loc[~pos, 'bust_size'] = np.nan
    # Split the number and the letter
    temp = df.bust_size[pos].str.extract(r'^(\d+)([A-K].*)$')
    df.loc[pos, 'bust_size'] = temp[0]
    df.bust_size = df.bust_size.astype(float, copy=False)
    df['cup_size'] = np.nan
    df.loc[pos, 'cup_size'] = temp[1]
    df.cup_size = df.cup_size.astype('category', copy=False)
    df.cup_size = df.cup_size.cat.set_categories(
        'AA A B C D D+ DD DDD/E F G H I J'.split(), ordered=True)
    return df


def body_type_to_cat(df: pd.DataFrame):
    '''
    Convert body_type to category.
    '''
    df.body_type = df.body_type.astype('category', copy=False)
    return df
  

def preprocess_age(df):
    '''
    Data preprocess: age column
    1. Convert all non-pure number item into NaN 
    2. Convert value >= 100 item into NaN\n
    After process:\n
    class: 84, dtype: int64
    '''
    df['age'] = df.age.astype(str, copy=False)
    # Convert all non-pure number item into NaN
    df['age'][~df.age.str.match(r'^[0-9]*$')] = np.nan
    # Convert value >= 100 item into NaN
    df['age'][df['age'].astype(float) >= 100] = np.nan

    return df
  
def preprocess_usually_wear(df):
    '''
    Data preprocess: age column\n
    Convert all non-pure number item into NaN 
    After process:\n
    class: 50, dtype: int64
    '''
    df['usually_wear'] = df.usually_wear.astype(str, copy=False)
    # Convert all non-pure number item into NaN
    df['usually_wear'][~df.usually_wear.str.match(r'^[0-9]*$')] = np.nan

    return df
  
# Maybe include all functions above in the future, making the preprocess procedure atomic commented by lai.
def preprocess(df: pd.DataFrame, is_train=False):
    '''
    Preprocess the data, Yunqin Zhu's version.
    '''
    df = arrange_cols(df, is_train)
    df = drop_ufeff(df)
    df = empty_to_nan(df)
    return df
