import wandb
import random
import pandas as pd
from sklearn.model_selection import train_test_split


def split_documents(data, config, seed=None):
    if seed is None:
        seed = random.randint(0, 2**30)
        wandb.config.update({"data.train_val_seed": seed})

    documents = data['file'].str.extract(r'(_.*_)').iloc[:, 0].str.replace('_', '')
    train_documents, test_documents = train_test_split(
        documents.value_counts().index.to_numpy(),
        test_size=config.test_size,
        random_state=seed,
        shuffle=True
    )
    return data[documents.isin(train_documents)], data[documents.isin(test_documents)]


def save_train_test_split(config):
    data = pd.read_excel(config.raw_path)
    data = data[data.fragment]
    train, test = split_documents(data, config, seed=42)
    train.to_csv(config.train_path)
    test.to_csv(config.test_path)

#%%
