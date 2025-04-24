import sys
import json
import logging
import pandas as pd
import torch


import CAT


from utils import dataset, CD_model_flag, device, CommonArgParser

args = CommonArgParser().parse_args()


def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


setuplogger()

train_triplets = pd.read_csv(f'./data/{dataset}/train_triples.csv', encoding='utf-8').to_records(index=False)
valid_triplets = pd.read_csv(f'./data/{dataset}/valid_triples.csv', encoding='utf-8').to_records(index=False)

concept_map = json.load(open(f'./data/{dataset}/concept_map.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
metadata = json.load(open(f'./data/{dataset}/metadata.json', 'r'))
train_data = CAT.dataset.TrainDataset(train_triplets, concept_map,
                                      metadata['num_train_students'],
                                      metadata['num_questions'],
                                      metadata['num_concepts'])

valid_data = CAT.dataset.TrainDataset(valid_triplets, concept_map,
                                      metadata['num_train_students'],
                                      metadata['num_questions'],
                                      metadata['num_concepts'])

if CD_model_flag == 'ncd':
    config = {
        'learning_rate': 0.001,
        'batch_size': 128,
        'num_epochs': 3,
        'device': device,
        'model_save_path': './model/ncd_pt/ncd_%s.pt' % dataset,
        'prednet_len1': 128,
        'prednet_len2': 64,
        'betas': (0.9, 0.999),
    }
    model = CAT.model.NCDModel(**config)

if CD_model_flag == 'mirt':
    config = {
        'learning_rate': 0.001,
        'batch_size': 128,
        'num_epochs': 5,
        'device': device,
        'model_save_path': './model/mirt_pt/mirt_%s.pt' % dataset,
        'dim': metadata['num_concepts'],
        'guess': False,
    }
    model = CAT.model.MIRTModel(**config)

model.init_model(train_data)
model.train(train_data, valid_data)
model.adaptest_save(model.config['model_save_path'])
