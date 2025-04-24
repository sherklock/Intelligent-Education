from collections import defaultdict, deque
import torch

try:
    from .dataset import Dataset
    from .train_dataset import TrainDataset
except (ImportError, SystemError):
    from dataset import Dataset
    from train_dataset import TrainDataset

class AdapTestDataset(Dataset):
    def __init__(self, data, concept_map, num_students, num_questions, num_concepts):
        super().__init__(data, concept_map, num_students, num_questions, num_concepts)
        self._tested = None
        self._untested = None
        self.student_ids = []
        self.reset()

    def apply_selection(self, student_idx, question_idx):
        assert question_idx in self._untested[student_idx], 'Selected question not allowed'
        self._untested[student_idx].remove(question_idx)
        self._tested[student_idx].append(question_idx)

    def reset(self):
        self._tested = defaultdict(deque)
        self._untested = defaultdict(set)
        self.student_ids = []
        for sid in self.data:
            self._untested[sid] = set(self.data[sid].keys())
            self.student_ids.append(sid)

    @property
    def tested(self):
        return self._tested

    @property
    def untested(self):
        return self._untested

    def get_tested_dataset(self, last=False, ssid=None):
        if ssid is None:
            triplets = []
            for sid, qids in self._tested.items():
                if last:
                    qid = qids[-1]
                    triplets.append((sid, qid, self.data[sid][qid]))
                else:
                    for qid in qids:
                        triplets.append((sid, qid, self.data[sid][qid]))
            return TrainDataset(triplets, self.concept_map, self.num_students, self.num_questions, self.num_concepts)
        else:
            triplets = []
            for sid, qids in self._tested.items():
                if ssid == sid:
                    if last:
                        qid = qids[-1]
                        triplets.append((sid, qid, self.data[sid][qid]))
                    else:
                        for qid in qids:
                            triplets.append((sid, qid, self.data[sid][qid]))
            return TrainDataset(triplets, self.concept_map, self.num_students, self.num_questions, self.num_concepts)