import CAT
import pandas as pd
import json
from utils import CommonArgParser, dataset
from draw import drawloss
import torch

torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)

args = CommonArgParser().parse_args()
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


best_loss = float('inf')
best_model_path = 'model/binary_code_ckt/'

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

if __name__ == "__main__":
    model = CAT.model.HashModel(**config)
    model.init_model(train_data)
    learning_rate = args.learning_rate
    print('learning rate: ', learning_rate)
    model_filename = f'{best_model_path}model4_epoch%s.pt' % dataset
    epoch_losses, avg_losses, aucs, accs = model.train(train_data, valid_data)
    model.save_snapshot(model_filename)
    drawloss(epoch_losses, avg_losses, aucs, accs)
