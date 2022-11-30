#!/usr/bin/env python3
'''
By running the script, the following files will be created:
- data/modcloth_data.json.gz        : ModCloth dataset from UCSD
- data/renttherunway_data.json.gz   : RentTheRunaway dataset from UCSD
- data/train_data.json              : Training data from MIRA Lab, modified from RentTheRunaway dataset
'''

dir_list = ['data']

file_list = [
    ('http://jmcauley.ucsd.edu/data/modcloth/modcloth_final_data.json.gz',
     'data/modcloth_data.json.gz'),
    ('http://jmcauley.ucsd.edu/data/renttherunway/renttherunway_final_data.json.gz',
     'data/renttherunway_data.json.gz'),
    ('http://miralab.ai/courses/ML2022Fall/project/train_data_all.json',
     'data/train_data.json'),
]

import os
from tqdm import tqdm


def make_dir(path, force=False, silent=False):
    '''
    Create a directory if it does not exist
    '''
    if not force and os.path.exists(path):
        if not silent:
            print(f'Directory "{path}" already exists')
        return
    if not silent:
        print(f'Create directory "{path}"')
    os.makedirs(path, exist_ok=True)


def download_file(url, out, force=False, silent=False):
    '''
    Download a file from a url to a given path, with a progress bar
    '''
    import requests
    if not force and os.path.exists(out):
        if not silent:
            print(f'File "{out}" already exists')
        return
    if not silent:
        print(f'Downloading {url} to "{out}"')
    response = requests.get(url, stream=True)
    length = int(response.headers.get('content-length', 0))
    with open(out, 'wb') as file:
        iterator = response.iter_content(chunk_size=1024)
        if not silent:
            iterator = tqdm(iterator, total=length // 1024, unit='KB')
        for data in iterator:
            file.write(data)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-f',
                        '--force',
                        action='store_true',
                        help='Force download')
    parser.add_argument('-s',
                        '--silent',
                        action='store_true',
                        help='Silent mode')
    args = parser.parse_args()

    for dir in dir_list:
        make_dir(dir, force=args.force, silent=args.silent)
    for url, out in file_list:
        download_file(url, out, force=args.force, silent=args.silent)
