import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import optim
from tqdm import tqdm

from data_loader import DatasLoader
from knowvae import ivae_train
from models import *
from util import build_exers_pool, convert_pool_to_tensor, counterfactual_sampling


class DbCD:
    def __init__(self, model, dataset, user_num, item_num, knowledge_num):
        super(DbCD, self).__init__()
        self.model = model
        self.dataset = dataset
        self.num_users = user_num
        self.num_items = item_num
        self.num_knowledge = knowledge_num
        self.u_mean = ...

        self.net = NCDNet(user_num, item_num, knowledge_num)

    def train(self, train_data, valid_data=None, item_knowledge=None, similar_pool=None, device="cpu",
              epoch=10, lr=0.001, alpha=0.1, gamma=0.01, lamb=0.0, u_mean=None, u_std=None) -> ...:
        self.net = self.net.to(device)
        item_knowledge = item_knowledge.to(device)
        self.u_mean = u_mean
        logging.info("Model Create Finish! Train Beginning!")

        eps = 1e-6
        self.net.train()

        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=lamb)

        x1, y1, y2, y3, y4 = [], [], [], [], []
        best_epoch, avg_s = 0, 0
        for epoch_i in range(1, epoch + 1):
            epoch_losses = []

            for batch_data in tqdm(train_data, desc="Epoch %s" % epoch_i):
                user_id, item_id, knowledge_emb, y = batch_data

                u_m_d = u_mean[user_id].to(device)
                u_s_d = u_std[user_id].to(device)
                x_sampling = counterfactual_sampling(item_id, similar_pool, device=device)
                k_emb_sampling = item_knowledge[x_sampling].to(device)
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)

                pred = self.net(user_id, item_id, knowledge_emb, mean=u_m_d, std=u_s_d, sample=True)
                factual_loss = loss_function(pred, y)

                pred_ul = self.net(user_id, x_sampling, k_emb_sampling, mean=u_m_d, std=u_s_d, sample=True)

                pred = pred.clamp(min=eps, max=1 - eps)
                pred_ul = pred_ul.clamp(min=eps, max=1 - eps)

                contrastive_loss = - (pred * pred_ul.log() + (1 - pred) * (1 - pred_ul).log())
                contrastive_loss = contrastive_loss.mean()

                confidence_penalty = (pred * pred.log() + (1 - pred) * (1 - pred).log()).mean()

                info_loss = alpha * contrastive_loss + gamma * confidence_penalty

                loss = factual_loss + info_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.4f" % (epoch_i, float(np.mean(epoch_losses))))
            x1.append(epoch_i)
            y1.append('%.4f' % float(np.mean(epoch_losses)))
            if valid_data is not None:
                auc, acc, rmse = self.eval(valid_data, device=device)
                print("[Epoch %d] acc: %.4f, auc: %.4f, rmse: %.4f" % (epoch_i, acc, auc, rmse))
                y2.append('%.4f' % acc)
                y3.append('%.4f' % auc)
                y4.append('%.4f' % rmse)
                if np.mean([acc, auc]) > avg_s:
                    avg_s = np.mean([acc, auc])
                    best_epoch = epoch_i
                    self.save(filepath=f'model/{self.model}-{self.dataset}')
        result_log = {'epoch': x1, 'loss': y1, 'acc': y2, 'auc': y3, 'rmse': y4}
        return best_epoch, result_log

    def eval(self, test_data, device='cpu'):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, desc="Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            u_m_d = self.u_mean[user_id].to(device)

            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)

            pred: torch.Tensor = self.net(user_id, item_id, knowledge_emb, mean=u_m_d, sample=False)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), rmse

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)


def main(model, dataset, epoch=10, lr=0.002, alpha=0.1, gamma=0.01, lamb=0.0, device='cpu'):
    cdm = DbCD(model, dataset, student_num, exer_num, knowledge_nums)
    best_epoch, result_log = cdm.train(train, valid, item_knowledge_embs, similar_pool_tensor, device=device,
                                       lr=lr, epoch=epoch, alpha=alpha, gamma=gamma, lamb=lamb,
                                       u_mean=stu_con_mean, u_std=stu_con_std,)
    result = pd.DataFrame(result_log)
    result.to_csv(f'result/{model}-{dataset}-epoch{epoch}_result', encoding='utf-8', index=False)
    cdm.load(filepath=f'model/{model}-{dataset}')
    # full_test
    auc, acc, rmse = cdm.eval(full_test, device=device)
    print("[after %d epochs]acc: %.4f, auc: %.4f, rmse: %.4f" % (best_epoch, acc, auc, rmse))
    full_test_result = {'acc': ['%.4f' % acc], 'auc': ['%.4f' % auc], 'rmse': ['%.4f' % rmse]}
    result = pd.DataFrame(full_test_result)
    result.to_csv(f'result/{model}-{dataset}-full_test_result', encoding='utf-8', mode='a', header=False, index=False)

    # random_test
    auc, acc, f1score = cdm.eval(random_test, device=device)
    print("[after %d epochs]acc: %.4f, auc: %.4f, rmse: %.4f" % (best_epoch, acc, auc, rmse))
    random_test_result = {'acc': ['%.4f' % acc], 'auc': ['%.4f' % auc], 'rmse': ['%.4f' % rmse]}
    result = pd.DataFrame(random_test_result)
    result.to_csv(f'result/{model}-{dataset}-random_test_result', encoding='utf-8', mode='a', header=False, index=False)

    # uniform_test
    auc, acc, rmse = cdm.eval(uniform_test, device=device)
    print("[after %d epochs]acc: %.4f, auc: %.4f, rmse: %.4f" % (best_epoch, acc, auc, rmse))
    uniform_test_result = {'acc': ['%.4f' % acc], 'auc': ['%.4f' % auc], 'rmse': ['%.4f' % rmse]}
    result = pd.DataFrame(uniform_test_result)
    result.to_csv(f'result/{model}-{dataset}-uniform_test_result', encoding='utf-8', mode='a', header=False, index=False)


if __name__ == '__main__':
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    threshold = 0.0

    dl = DatasLoader(dataset='a0910', batch_size=128)
    train, valid, full_test, random_test, uniform_test, student_num, exer_num, knowledge_nums, item_knowledge_embs = dl.loader_data()

    train_matrix, train_mask, val_matrix, val_mask, item_diff = dl.construct_vae_dataset(train_ratio=0.9)
    vae_model = ivae_train(exer_num, knowledge_nums, device=DEVICE)
    vae_model.train_eval(train_matrix, train_mask, val_matrix, val_mask, item_diff, item_knowledge_embs)

    full_matrix, full_mask, _, _, item_diff = dl.construct_vae_dataset(train_ratio=1)
    vae_model.save_vae_params(full_matrix, full_mask, item_diff, item_knowledge_embs, batch_size=256)

    stu_con_mean = torch.load("mean.pt")
    stu_con_std = torch.load("std.pt")

    exers_pools, max_similar_num = build_exers_pool(item_knowledge_embs, threshold=threshold)
    similar_pool_tensor = convert_pool_to_tensor(exers_pools, max_similar_num).to(DEVICE)

    logging.getLogger().setLevel(logging.INFO)

    main('ncd', 'a0910', epoch=10, lr=0.002, lamb=0, device=DEVICE)
