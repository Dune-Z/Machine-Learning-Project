#!/usr/bin/env python3
'''
Main script.
'''

from preprocess import Preprocess
PATH = '../../train_data_all.json'


def main():
    preprocess = Preprocess(PATH)
    preprocess.preprocess(is_train=False)
    df = preprocess.df


if __name__ == '__main__':
    main()
