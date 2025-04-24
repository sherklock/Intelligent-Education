import logging
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from CAT.model.abstract_model import BinaryAbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from utils import CommonArgParser, dataset, device
from utils import early_stop, check_overfitting
from collections import defaultdict
import time

args = CommonArgParser().parse_args()

print("device: ", device)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        hidden = F.relu(self.encoder(x))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon_output = torch.sigmoid(self.decoder(z))
        return z, mu, logvar, recon_output

class STE_XOR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_code, question_code):
        ctx.save_for_backward(student_code, question_code)
        return (student_code + question_code) % 2

    @staticmethod
    def backward(ctx, grad_output):
        student_code, question_code = ctx.saved_tensors
        grad_student = grad_output * (1 - 2 * question_code)
        grad_question = grad_output * (1 - 2 * student_code)
        return grad_student, grad_question

class BinaryCode(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, student_dim, question_dim, low_dim, batch_size):
        super(BinaryCode, self).__init__()
        self.exer_n = exer_n
        self.student_n = student_n
        self.knowledge_n = knowledge_n
        self.student_dim = student_dim
        self.question_dim = question_dim
        self.lowdim = low_dim
        self.batch_size = batch_size
        self.attetion_embed_dim = self.knowledge_n
        self.student_emb = nn.Embedding(self.student_n, self.lowdim)
        self.knowledge_emb = nn.Embedding(self.knowledge_n, self.lowdim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.lowdim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.k_index = torch.LongTensor(list(range(self.knowledge_n))).to(device)
        self.prednet_input_len = self.knowledge_n
        self.prednet_full1 = nn.Linear(self.prednet_input_len, 128)
        self.drop_1 = nn.Dropout(0.5)
        self.prednet_full2 = nn.Linear(128, 64)
        self.drop_2 = nn.Dropout(0.5)
        self.prednet_full3 = nn.Linear(64, 1)
        self.student_vae = VAE(input_dim=self.knowledge_n, hidden_dim=64, latent_dim=self.knowledge_n)
        self.question_vae = VAE(input_dim=self.knowledge_n, hidden_dim=64, latent_dim=self.knowledge_n)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        self.tau = 100
        self.alpha = 0.5
        self.beta = 0.7
        self.VaeLossAlpha = 0.1
        self.ConsisenceLossAlpha = 0.3
        self.num_hashes = 10
        self.w = 0.5
        self.init_lsh_params()

    def init_lsh_params(self):
        self.a = torch.randn(self.num_hashes, self.knowledge_n, device=device)
        self.c = torch.rand(self.num_hashes, 1, device=device)
        self.random_dims = torch.randint(0, self.knowledge_n, (self.num_hashes,), device=device)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def binary_gumbel_softmax(self, logits):
        logits = logits.unsqueeze(-1)
        logits = torch.cat([logits, torch.zeros_like(logits)], dim=-1)
        gumbel_out = F.gumbel_softmax(logits, tau=self.tau, hard=True)
        return gumbel_out[:, :, 0]

    def LSH_Hash_Real(self, embedding, num_hashes=None, w=None):
        if num_hashes is None:
            num_hashes = self.num_hashes
        if w is None:
            w = self.w
        N, D = embedding.shape
        a = self.a[:num_hashes, :D]
        c = self.c[:num_hashes]
        hash_codes = torch.floor((torch.matmul(embedding, a.T) + c.T) / w)
        return hash_codes

    def LSH_Hash_Binary(self, binary_embeddings, num_hashes=None):
        if num_hashes is None:
            num_hashes = self.num_hashes
        random_dims = self.random_dims[:num_hashes]
        hash_codes = torch.index_select(binary_embeddings, 1, random_dims)
        return hash_codes

    def Bulid_LSH_Similarity_Matrix(self, hash_codes):
        similarity_matrix = (hash_codes.unsqueeze(1) == hash_codes.unsqueeze(0)).all(dim=2).float()
        return similarity_matrix

    def LSH_Consistency_Loss(self, real_embeddings, binary_embeddings, real_similarity_matrix, binary_similarity_matrix, hash_codes):
        N = real_embeddings.size(0)
        D = real_embeddings.size(1)
        real_diff = real_embeddings.unsqueeze(1) - real_embeddings.unsqueeze(0)
        real_loss = torch.sum(real_diff.pow(2) * binary_similarity_matrix.unsqueeze(-1), dim=2)
        real_loss = torch.sum(real_loss) / (N * N)
        binary_sim = torch.matmul(binary_embeddings, binary_embeddings.T)
        binary_loss = torch.sum(binary_sim * real_similarity_matrix) / (N * N)
        total_loss = real_loss + binary_loss
        return total_loss

    def consistency_loss(self, real_students, binary_students, real_questions, binary_questions):
        student_real_hash = self.LSH_Hash_Real(real_students)
        student_binary_hash = self.LSH_Hash_Binary(binary_students)
        question_real_hash = self.LSH_Hash_Real(real_questions)
        question_binary_hash = self.LSH_Hash_Binary(binary_questions)
        student_real_similarity = self.Bulid_LSH_Similarity_Matrix(student_real_hash)
        student_binary_similarity = self.Bulid_LSH_Similarity_Matrix(student_binary_hash)
        question_real_similarity = self.Bulid_LSH_Similarity_Matrix(question_real_hash)
        question_binary_similarity = self.Bulid_LSH_Similarity_Matrix(question_binary_hash)
        student_loss = self.LSH_Consistency_Loss(real_students, binary_students, student_real_similarity, student_binary_similarity, student_real_hash)
        question_loss = self.LSH_Consistency_Loss(real_questions, binary_questions, question_real_similarity, question_binary_similarity, question_real_hash)
        return student_loss + question_loss

    def forward(self, student_idx=None, exer_idx=None, kn_emb=None, test_mode=False):
        exer_idx = exer_idx.to(self.k_difficulty.weight.device)
        knowledge_low_emb = self.knowledge_emb(self.k_index.to(self.knowledge_emb.weight.device))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_idx.to(self.e_discrimination.weight.device))) * 10
        if not test_mode:
            batch_student_emb = self.student_emb(student_idx)
            batch_student_emb = torch.sigmoid(torch.mm(batch_student_emb, knowledge_low_emb.T))
            student_latent, student_mu, student_logvar, student_recon = self.student_vae(batch_student_emb)
            student_binary_code = self.binary_gumbel_softmax(student_latent)
        else:
            student_binary_code = None
        batch_question_emb = self.k_difficulty(exer_idx)
        batch_question_emb = torch.sigmoid(torch.mm(batch_question_emb, knowledge_low_emb.T))
        question_latent, question_mu, question_logvar, question_recon = self.question_vae(batch_question_emb)
        question_binary_code = self.binary_gumbel_softmax(question_latent)
        if test_mode:
            return question_binary_code, question_latent, e_discrimination
        ability_difference = STE_XOR.apply(student_binary_code, question_binary_code)
        ability_difference = ability_difference.float().mean(dim=1, keepdim=True)
        matching = (student_binary_code * question_binary_code).float().mean(dim=1, keepdim=True)
        predicted_ability = e_discrimination * (self.alpha * (1 - ability_difference) + self.beta * matching) * kn_emb
        student_reconstruction_loss = F.mse_loss(student_recon, batch_student_emb, reduction='mean')
        student_kl_loss = -0.5 * torch.sum(1 + student_logvar - student_mu.pow(2) - student_logvar.exp())
        question_reconstruction_loss = F.mse_loss(question_recon, batch_question_emb, reduction='mean')
        question_kl_loss = -0.5 * torch.sum(1 + question_logvar - question_mu.pow(2) - question_logvar.exp())
        vae_loss = student_reconstruction_loss + student_kl_loss + question_reconstruction_loss + question_kl_loss
        consistency_loss = self.consistency_loss(real_students=student_latent, binary_students=student_binary_code, real_questions=question_latent, binary_questions=question_binary_code)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(predicted_ability)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        return output, vae_loss, consistency_loss

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class HashModel(BinaryAbstractModel):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None

    @property
    def name(self):
        return 'Hash Code Pre-Train(vea-consistency)'

    def init_model(self, data: Dataset):
        self.model = BinaryCode(
            student_n=self.config['student_n'],
            exer_n=self.config['exer_n'],
            knowledge_n=self.config['knowledge_n'],
            student_dim=self.config['student_dim'],
            question_dim=self.config['question_dim'],
            low_dim=self.config['low_dim'],
            batch_size=self.config['batch_size']
        )

    def train(self, train_data: TrainDataset, valid_data: TrainDataset):
        model_name = self.name
        print(f"Training Name:{model_name}")
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        early = 3
        overfit_threshold = 0.02
        self.model.to(device)
        logging.info('train on {}'.format(device))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        epoch_losses = {}
        avg_losses = []
        aucs = []
        accs = []
        metrics_log = defaultdict(list)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_name = f"model4vae-consistency-traintime-{dataset}-{timestamp}.txt"
        log_file_path = os.path.join(current_dir, '../../binary model result', log_file_name)
        alpha = getattr(self.model, 'alpha', 'Undefined')
        beta = getattr(self.model, 'beta', 'Undefined')
        tau = getattr(self.model, 'tau', 'Undefined')
        num_hashes = getattr(self.model, 'num_hashes', 'Undefined')
        w = getattr(self.model, 'w', 'Undefined')
        VaeLossAlpha = getattr(self.model, 'VaeLossAlpha', 'Undefined')
        ConssistenceLossAlpha = getattr(self.model, 'ConsisenceLossAlpha', 'Undefined')
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Training Name: {model_name}\n")
            log_file.write(f"Using DataSet: {dataset}\n")
            log_file.write(f"Learning Rate: {lr}\n")
            log_file.write(f"Batch Size: {batch_size}\n")
            log_file.write(f"Epochs: {epochs}\n")
            log_file.write(f"Gumbel-softmax temp: {tau}\n")
            log_file.write(f"Alpha: {alpha}\n")
            log_file.write(f"Beta: {beta}\n")
            log_file.write(f"VaeLossAlpha: {VaeLossAlpha}\n")
            log_file.write(f"num_hashes: {num_hashes}\n")
            log_file.write(f"w: {w}\n")
            for ep in range(1, epochs + 1):
                start_time = time.time()
                loss = []
                for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(train_loader):
                    student_ids = student_ids.to(device)
                    question_ids = question_ids.to(device)
                    concepts_emb = concepts_emb.to(device)
                    labels = labels.to(device)
                    pred, vaeloss, consistency_loss = self.model(student_ids, question_ids, concepts_emb)
                    task_loss = self._loss_function(pred, labels)
                    total_loss = task_loss + VaeLossAlpha * vaeloss + ConssistenceLossAlpha * consistency_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    self.model.apply_clipper()
                    loss.append(total_loss.mean().item())
                epoch_loss = np.mean(loss)
                epoch_time = time.time() - start_time
                print(f"[Epoch {ep}] LogisticLoss: {epoch_loss:.6f}, Train Time: {epoch_time:.4f} seconds")
                epoch_losses[ep] = epoch_loss
                avg_loss, auc, acc = self.validata(valid_data)
                print("Validation Loss: %.6f, AUC: %.4f, Accuracy: %.4f" % (avg_loss, auc, acc))
                avg_losses.append(avg_loss)
                aucs.append(auc)
                accs.append(acc)
                metrics_log['loss'].append(avg_loss)
                metrics_log['auc'].append(auc)
                metrics_log['acc'].append(acc)
                log_file.write(f"Epoch {ep}:\n")
                log_file.write(f"  Training Loss: {epoch_loss:.6f}\n")
                log_file.write(f"  Validation Loss: {avg_loss:.6f}\n")
                log_file.write(f"  AUC: {auc:.4f}\n")
                log_file.write(f"  Accuracy: {acc:.4f}\n\n")
                log_file.write(f"  Time: {epoch_time:.4f} seconds\n\n")
                if ep > 1 and check_overfitting(metrics_log, 'auc', overfit_threshold, show=True):
                    print("Early stopping due to overfitting!")
                    break
                early = early_stop(metrics_log['auc'], early, threshold=0)
                if early <= 0:
                    print("Early stopping!")
                    break
        return epoch_losses, avg_losses, aucs, accs

    def validata(self, val_data: TrainDataset):
        device = self.config['device']
        val_loader = DataLoader(val_data, batch_size=self.config['batch_size'], shuffle=True)
        all_preds = []
        all_labels = []
        loss = []
        with torch.no_grad():
            self.model.eval()
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(val_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred, _, _ = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                loss.append(bz_loss.mean().item())
                all_preds += pred.tolist()
                all_labels += labels.tolist()
            self.model.train()
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        avg_loss = np.mean(loss)
        auc = roc_auc_score(all_labels, all_preds)
        threshold = 0.5
        binary_pred = (all_preds >= threshold).astype(int)
        acc = accuracy_score(all_labels, binary_pred)
        return avg_loss, auc, acc

    def _loss_function(self, pred, real):
        pred_0 = torch.ones(pred.size()).to(self.config['device']) - pred
        output = torch.cat((pred_0, pred), 1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)

    def adaptest_save(self, path):
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'student' not in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])

    def adaptest_update(self, adaptest_data: AdapTestDataset):
        alpha = getattr(self.model, 'alpha', 'Undefined')
        beta = getattr(self.model, 'beta', 'Undefined')
        tau = getattr(self.model, 'tau', 'Undefined')
        VaeLossAlpha = getattr(self.model, 'VaeLossAlpha', 'Undefined')
        ConssistenceLossAlpha = getattr(self.model, 'ConsisenceLossAlpha', 'Undefined')
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.student_emb.parameters(), lr=lr)
        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)
        for ep in range(1, epochs + 1):
            loss = 0.0
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device)
                concepts_emb = concepts_emb.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                pred, vaeloss, consistency_loss = self.model(student_ids, question_ids, concepts_emb)
                task_loss = self._loss_function(pred, labels)
                total_loss = task_loss + VaeLossAlpha * vaeloss + ConssistenceLossAlpha * consistency_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += total_loss.data.float()
        return loss

    def evaluate(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']
        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.FloatTensor(concepts_embs).to(device)
                output, _, _ = self.model(student_ids, question_ids, concepts_embs)
                output = output.view(-1)
                pred += output.tolist()
            self.model.train()
        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts.update(set(concept_map[qid]))
            for qid in adaptest_data.tested[sid]:
                tested_concepts.update(set(concept_map[qid]))
            converage = len(tested_concepts) / len(all_concepts)
            coverages.append(converage)
        cov = sum(coverages) / len(coverages)
        real = np.array(real)
        real = np.where(real > 0.5, 1.0, 0.0)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)
        threshold = 0.5
        binary_pred = (pred >= threshold).astype(int)
        acc = accuracy_score(real, binary_pred)
        return {
            'auc': auc,
            'cov': cov,
            'acc': acc
        }

    def get_knowledge_status(self, stu_id):
        stu_id = stu_id.to(self.config['device'])
        stu_low_emb = self.model.student_emb(stu_id)
        stu_low_emb = stu_low_emb.unsqueeze(0)
        knowledge_low_emb = self.model.knowledge_emb(self.model.k_index.to(self.model.knowledge_emb.weight.device))
        stu_emb = torch.sigmoid(torch.mm(stu_low_emb, knowledge_low_emb.T))
        student_latent, student_mu, student_logvar, student_recon = self.model.student_vae(stu_emb)
        student_binary_code = self.model.binary_gumbel_softmax(student_latent)
        return stu_emb.data, student_binary_code.data

    def get_pred(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']
        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    def save_snapshot(self, path):
        model_dict = self.model.state_dict()
        torch.save(model_dict, path)

    def load_snapshot(self, path):
        f = open(path, 'rb')
        self.model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
        f.close()