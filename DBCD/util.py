import numpy as np
import torch

from scipy.sparse import csr_matrix


def counterfactual_sampling(i_ids, pool_tensor, device):
    B = i_ids.shape[0]
    max_k = pool_tensor.shape[1]
    candidate_pool = pool_tensor[i_ids]
    invalid_mask = candidate_pool == -1
    sampled = torch.full((B,), -1, dtype=torch.long, device=device)

    for i in range(B):
        valid = candidate_pool[i][~invalid_mask[i]]
        if len(valid) > 0:
            sampled[i] = valid[torch.randint(0, len(valid), (1,))]
        else:
            sampled[i] = i_ids[i]
    return sampled


def build_exers_pool(q, threshold: float = 0.1):
    if isinstance(q, torch.Tensor):
        q = q.cpu().numpy()

    q_sparse = csr_matrix(q)
    intersection = q_sparse @ q_sparse.T  # [I, I]
    item_counts = np.array(q_sparse.sum(axis=1)).reshape(-1)  # [I]
    union = item_counts[:, None] + item_counts[None, :] - intersection.toarray()
    union = np.clip(union, 1e-8, None)

    similarity = intersection.toarray() / union
    similarity[similarity <= threshold] = 0.0

    similar_pool = {}
    max_similar_num = 0
    for i in range(similarity.shape[0]):
        similar = np.where(similarity[i] > 0)[0].tolist()
        if i in similar:
            similar.remove(i)
        similar_pool[i] = similar
        max_similar_num = max(max_similar_num, len(similar))

    return similar_pool, max_similar_num


def convert_pool_to_tensor(pool_dict, max_k):
    num_items = len(pool_dict)
    pool_tensor = torch.full((num_items, max_k), -1, dtype=torch.long)

    for i, neighbors in pool_dict.items():
        top_k = neighbors[:max_k]
        pool_tensor[i, :len(top_k)] = torch.tensor(top_k, dtype=torch.long)

    return pool_tensor
