from preprocess import Preprocess
from xgboost import XGBClassifier
PATH = '../../train_data_all.json'


def main():
    preprocess = Preprocess(PATH)
    preprocess.preprocess(is_train=False)
    df = preprocess.df
    model = XGBClassifier()


if __name__ == '__main__':
    main()
