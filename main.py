import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from omegaconf import OmegaConf
from models.transformer.train_transformer import train_transformer
from utils.initializers import initialize_data
import random
import hydra
import wandb
import yaml
import numpy as np
import pandas as pd
import torch
import nltk
import transformers
from datasets import load_metric
from transformers import Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, AutoTokenizer, Text2TextGenerationPipeline, Seq2SeqTrainer
from os.path import exists
from models.baseline.baseline import Baseline
from models.transformer.dataset import EllipsesDataset
from models.transformer.predict_transformer import predict_transformer


def main():

    os.chdir("/home/Niklas.Kaemmer/AML4DH")
    predict_transformer()

    """with open('configs/config.yaml', 'r') as file:
        config_file = yaml.safe_load(file)


    wandb.init(entity="nkaem", project="ellipses-resolution", config=config_file)
    wandb.init(entity="nkaem", project="ellipses-resolution")
    config = wandb.config

    random.seed(config.random_gen_seed)
    os.chdir(config.project_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.model_name == "facebook/mbart-large-50-many-to-many-mmt":
        tokenizer.src_lang = 'de_DE'
        tokenizer.tgt_lang = 'de_DE'
    if config.model_name == "facebook/m2m100_418M":
        tokenizer.src_lang = 'de'
        tokenizer.tgt_lang = 'de'

    train_data, val_data = initialize_data(config, tokenizer, mode='training')

    train_transformer(train_data, val_data, tokenizer, config)
    wandb.finish()"""


if __name__ == "__main__":

    print("===== CUDA AVAILABLE ======")
    print(torch.cuda.is_available())
    print("===========================")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#%%
