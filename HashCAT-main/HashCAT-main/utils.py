import sys
import os
import torch
import numpy as np
import argparse
import json

sys.path.append(os.path.dirname(sys.path[0]))

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

dataset = "nips"
CD_model_flag = 'mirt'
print("using dataset: ", dataset)
print("using CD model: ", CD_model_flag)
metadata_path = f'./data/{dataset}/metadata.json'
with open(metadata_path, 'r') as file:
    jsondata = json.load(file)

num_concepts = jsondata['num_concepts']
num_students = jsondata['num_students']
num_questions = jsondata['num_questions']

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=num_questions, help='The number of exercise')
        self.add_argument('--knowledge_n', type=int, default=num_concepts, help='The number of concepts')
        self.add_argument('--student_n', type=int, default=num_students, help='The number of students')
        self.add_argument('--batch_size', type=int, default=1024, help='The size of batch')
        self.add_argument('--epochs', type=int, default=14, help='The number of epochs')
        self.add_argument('--learning_rate', type=float, default=0.0005, help='The learning rate')
        self.add_argument('--low_dim', type=int, default=20, help='The dimension of knowledge embedding dim')
        self.add_argument('--student_dim', type=int, default=num_concepts, help='The dimension of student binary_code')
        self.add_argument('--question_dim', type=int, default=num_concepts, help='The dimension of exercise binary_code')
        self.add_argument('--device', type=str, default=device, help='CPU or GPU')

def get_perf(metrics_log, window_size, target, show=True):
    maxs = {title: 0 for title in metrics_log.keys()}
    assert target in maxs
    length = len(metrics_log[target])
    for v in metrics_log.values():
        assert length == len(v)
    if window_size >= length:
        for k, v in metrics_log.items():
            maxs[k] = np.mean(v)
    else:
        for i in range(length-window_size):
            now = np.mean(metrics_log[target][i:i+window_size])
            if now > maxs[target]:
                for k, v in metrics_log.items():
                    maxs[k] = np.mean(v[i:i+window_size])
    if show:
        for k, v in maxs.items():
            print('{}:{:.5f}'.format(k, v), end=' ')
    return maxs

def check_overfitting(metrics_log, target, threshold=0.02, show=False):
    maxs = get_perf(metrics_log, 1, target, False)
    assert target in maxs
    overfit = (maxs[target]-metrics_log[target][-1]) > threshold
    if overfit and show:
        print('***********overfit*************')
        print('best:', end=' ')
        for k, v in maxs.items():
            print('{}:{:.5f}'.format(k, v), end=' ')
        print('')
        print('now:', end=' ')
        for k, v in metrics_log.items():
            print('{}:{:.5f}'.format(k, v[-1]), end=' ')
        print('')
        print('***********overfit*************')
    return overfit

def early_stop(metric_log, early, threshold=0.01):
    if len(metric_log) >= 2 and metric_log[-1] < metric_log[-2] and metric_log[-1] > threshold:
        return early - 1
    else:
        return early