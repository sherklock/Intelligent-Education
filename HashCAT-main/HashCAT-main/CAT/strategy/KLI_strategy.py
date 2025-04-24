import numpy as np
import time
import torch
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset

class KLIStrategy(AbstractStrategy):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Kullback-Leibler Information Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'get_kli'), 'the models must implement get_kli method'
        assert hasattr(model, 'get_pred'), 'the models must implement get_pred method for accelerating'
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        total_time = 0
        test_num_students = len(adaptest_data.student_ids)
        for sid in adaptest_data.student_ids:
            n = len(adaptest_data.tested[sid])
            if model.name == 'Neural Cognitive Diagnosis':
                theta = model.get_knowledge_status(torch.tensor(sid, dtype=torch.long))
            else:
                theta = model.get_knowledge_status(torch.tensor(sid, dtype=torch.long))
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            start_time = time.time()
            untested_kli = [model.get_kli(sid, qid, n, pred_all) for qid in untested_questions]
            j = np.argmax(untested_kli)
            selection[sid] = untested_questions[j]
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            print(f'Kullback-Leibler Information Strategy: {sid}th student, select question: {untested_questions[j]}, cost time: {execution_time}')
        av_time = total_time / test_num_students
        return selection, total_time, av_time

class MKLIStrategy(KLIStrategy):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Multivariate Kullback-Leibler Information Strategy'