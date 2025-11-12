import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import TensorDataset, DataLoader


class DatasLoader:
    def __init__(self, dataset, batch_size=128):
        self.data_set = dataset
        self.batch_size = batch_size
        self.user_n = ...
        self.exer_n = ...

    def loader_data(self):
        train_data = pd.read_csv(f"data/{self.data_set}/train.csv")
        valid_data = pd.read_csv(f"data/{self.data_set}/valid.csv")
        full_test_data = pd.read_csv(f"data/{self.data_set}/full_test.csv")
        random_test_data = pd.read_csv(f"data/{self.data_set}/random_test.csv")
        uniform_test_data = pd.read_csv(f"data/{self.data_set}/uniform_test.csv")
        q_matrix = pd.read_csv(f"data/{self.data_set}/Q.csv")

        # 读取Q矩阵内容
        item_knowledge = {}
        knowledge_set = set()
        for i, s in q_matrix.iterrows():
            item_id, knowledge_codes = s['exer_id'], list(set(eval(s['knowledge_code'])))
            item_knowledge[item_id] = knowledge_codes
            knowledge_set.update(knowledge_codes)

        user_n = np.max(train_data['user_id'])
        item_n = np.max([np.max(train_data['exer_id']), np.max(valid_data['exer_id']), np.max(full_test_data['exer_id'])])
        knowledge_n = np.max(list(knowledge_set))

        self.user_n = user_n
        self.exer_n = item_n

        q = torch.zeros((item_n, knowledge_n))
        for idx in range(item_n):
            q[idx][np.array(item_knowledge[idx + 1]) - 1] = 1.0

        train_set, valid_set, full_test_set, random_test_set, uniform_test_set = [
            transform(data["user_id"], data["exer_id"], knowledge_n, item_knowledge, data["score"], self.batch_size)
            for data in [train_data, valid_data, full_test_data, random_test_data, uniform_test_data]
        ]
        return train_set, valid_set, full_test_set, random_test_set, uniform_test_set, user_n, item_n, knowledge_n, q

    def construct_vae_dataset(self, train_ratio, seed=1234):
        train_data = pd.read_csv(f"data/{self.data_set}/train.csv")
        valid_data = pd.read_csv(f"data/{self.data_set}/valid.csv")
        exer_priori = pd.read_csv(f"data/{self.data_set}/exer_priori.csv")
        df = pd.concat([train_data, valid_data])

        n_users = self.user_n
        n_items = self.exer_n
        shape = (n_users, n_items)

        exer_priori['diff_dis'] = 1.0 - exer_priori['correct_rate']

        difficulty_vector = np.zeros((n_items,))
        for _, row in exer_priori.iterrows():
            exer_id = int(row['exer_id']) - 1
            difficulty_vector[exer_id] = row['diff_dis']

        if train_ratio == 1:
            matrix = df_to_csr(df, shape)
            mask = df_to_mask_csr(df, shape)
            return matrix.toarray(), mask.toarray(), None, None, difficulty_vector

        np.random.seed(seed)
        all_user_index = np.arange(n_users)
        n_train_users = int(train_ratio * n_users)

        train_user_index = np.sort(np.random.choice(all_user_index, size=n_train_users, replace=False))
        test_user_index = np.setdiff1d(all_user_index, train_user_index)

        full_matrix = df_to_csr(df, shape)
        full_mask = df_to_mask_csr(df, shape)

        train_matrix = full_matrix[train_user_index]
        train_mask = full_mask[train_user_index]

        val_matrix = full_matrix[test_user_index]
        val_mask = full_mask[test_user_index]

        return (train_matrix.toarray(), train_mask.toarray(),
                val_matrix.toarray(), val_mask.toarray(), difficulty_vector)


def transform(user_ids, item_ids, knowledge_n, item_knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item_ids), knowledge_n))
    for idx in range(len(item_ids)):
        knowledge_emb[idx][np.array(item_knowledge[item_ids[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user_ids, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item_ids, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def df_to_csr(df, shape=None):
    rows = df["user_id"] - 1
    cols = df["exer_id"] - 1
    values = df["score"]
    if shape is None:
        n_users = df["user_id"].max()
        n_items = df["exer_id"].max()
        shape = (n_users, n_items)
    mat = csr_matrix((values, (rows, cols)), shape=shape)
    return mat


def df_to_mask_csr(df, shape=None):
    rows = df["user_id"] - 1
    cols = df["exer_id"] - 1
    values = np.ones_like(df["score"])
    if shape is None:
        n_users = df["user_id"].max()
        n_items = df["exer_id"].max()
        shape = (n_users, n_items)
    mat = csr_matrix((values, (rows, cols)), shape=shape)
    return mat
