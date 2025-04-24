import torch
from torch.utils import data

try:
    # for python module
    from .dataset import Dataset
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from dataset import Dataset


class TrainDataset(Dataset, data.dataset.Dataset):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        super().__init__(data, concept_map,
                         num_students, num_questions, num_concepts)

    def __getitem__(self, item):
        sid, qid, score = self.raw_data[item]
        concepts = self.concept_map[qid]
        concepts_emb = [0.] * self.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        return sid, qid, torch.Tensor(concepts_emb), score

    def __len__(self):
        return len(self.raw_data)
