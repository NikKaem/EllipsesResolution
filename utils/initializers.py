from os.path import exists
import pandas as pd

from models.transformer.dataset import EllipsesDataset
from utils.split_dataset import save_train_test_split, split_documents


def initialize_data(config, tokenizer, mode='training'):
    if not exists(config.train_path) or not exists(config.test_path):
        print('Split data not found. Creating new split...')
        save_train_test_split(config)

    if mode == 'training':
        train_data = pd.read_csv(config.train_path)
        train, val = split_documents(train_data, config)
        return EllipsesDataset(train.raw_sentence, train.full_resolution, tokenizer), \
               EllipsesDataset(val.raw_sentence, val.full_resolution, tokenizer)

    elif mode == 'inference':
        test = pd.read_csv(config.test_path)
        return EllipsesDataset(test.raw_sentence, test.full_resolution, tokenizer)

    else:
        raise ValueError(f'no mode with name {mode}')
