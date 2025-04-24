import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import json
import datetime
import logging
import torch
import numpy as np
import pandas as pd
import CAT
from utils import CD_model_flag, dataset, device

logging.basicConfig(level=logging.INFO)
seed = 25
np.random.seed(seed)
torch.manual_seed(seed)

testdata = pd.read_csv(f'./data/{dataset}/test_triples.csv')
ckpt_path = './model/%s_pt/%s_%s.pt' % (CD_model_flag, CD_model_flag, dataset)

test_triplets = pd.read_csv(f'./data/{dataset}/test_triples.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'./data/{dataset}/concept_map.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
metadata = json.load(open(f'./data/{dataset}/metadata.json', 'r'))
test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                        metadata['num_test_students'], metadata['num_questions'],
                                        metadata['num_concepts'])
config = {
    'learning_rate': 0.0025,
    'batch_size': 1024,
    'num_epochs': 8,
    'num_dim': 1,
    'device': device,
    'prednet_len1': 128,
    'prednet_len2': 64,
    'THRESHOLD': 300,
    'start': 0,
    'end': 3000,
    'policy': 'notbobcat',
    'betas': (0.9, 0.999),
    'policy_path': 'policy.pt',
    'dim': metadata['num_concepts'],
    'guess': False,
}

if CD_model_flag == 'ncd':
    model = CAT.model.NCDModel(**config)
if CD_model_flag == 'irt':
    model = CAT.model.IRTModel(**config)
if CD_model_flag == 'mirt':
    model = CAT.model.MIRTModel(**config)

test_length = 20
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
results_path = f'./results/CAT/{now}'
strategies = [CAT.strategy.HashMABUCBStrategy()]

for strategy in strategies:
    model.init_model(test_data)
    model.adaptest_load(ckpt_path)
    test_data.reset()
    CAT.AdapTestDriver.run(model, strategy, test_data, test_length, results_path, dataset)