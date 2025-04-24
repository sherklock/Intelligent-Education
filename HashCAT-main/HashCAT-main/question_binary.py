from torch.utils.data import DataLoader, TensorDataset
import torch
import json
import os
import numpy as np
from utils import CommonArgParser
import pandas as pd
from utils import dataset
import CAT

args = CommonArgParser().parse_args()
test_triplets = pd.read_csv(f'./data/{dataset}/test_triples.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'./data/{dataset}/concept_map.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
metadata = json.load(open(f'./data/{dataset}/metadata.json', 'r'))
test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                        metadata['num_test_students'], metadata['num_questions'],
                                        metadata['num_concepts'])
config = {
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
    'num_epochs': args.epochs,
    'device': args.device,
    'student_dim': args.student_dim,
    'question_dim': args.question_dim,
    'low_dim': args.low_dim,
    'student_n': args.student_n,
    'exer_n': args.exer_n,
    'knowledge_n': args.knowledge_n,
}

model = CAT.model.HashModel(**config)
model.init_model(test_data)
json_path = os.path.join('CAT/strategy/QuestionBinaryBank', f'{dataset}_question_binary4.json')

def get_question_binary(model, batch_size=1024):
    all_question_ids = [item[1] for item in test_triplets]
    all_question_ids = torch.LongTensor(all_question_ids).to(device=args.device)
    question_dataset = TensorDataset(all_question_ids)
    question_loader = DataLoader(question_dataset, batch_size=batch_size, shuffle=False)
    model.load_snapshot('model/binary_code_ckt/model4_%s.pt' % dataset)
    model.model.eval()
    question_data_dict = {}
    with torch.no_grad():
        for batch in question_loader:
            exer_idx = batch[0]
            question_binary_code, batch_question_emb, e_discrimination = model.model(
                student_idx=None,
                exer_idx=exer_idx,
                test_mode=True
            )
            for qid, qbc, diff, discrim in zip(
                    exer_idx, question_binary_code, batch_question_emb, e_discrimination
            ):
                qid = qid.item()
                mean_diff = round(diff.mean().cpu().item(), 4)
                discrim = [round(d, 4) for d in discrim.cpu().numpy().tolist()]
                qbc = torch.nan_to_num(qbc, nan=0.0, posinf=1.0, neginf=0.0)
                question_data_dict[qid] = {
                    'binary_code': qbc.cpu().numpy().astype(int).tolist(),
                    'difficulty': mean_diff,
                    'discrimination': discrim
                }
    save_to_json(question_data_dict, json_path)

def save_to_json(data, filename):
    numerical_data = {int(key): value for key, value in data.items()}
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    existing_data.update(numerical_data)
    with open(filename, 'w') as f:
        json.dump(existing_data, f, separators=(',', ':'))

if __name__ == '__main__':
    get_question_binary(model)
    print("question binary code completely!")