import numpy as np
import time
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class MFIStrategy(AbstractStrategy):
    def __init__(self):
        super().__init__()
        self.I = None

    @property
    def name(self):
        return 'Maximum Fisher Information Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'get_fisher'), \
            'the models must implement get_fisher method'
        assert hasattr(model, 'get_pred'), \
            'the models must implement get_pred method for accelerating'
        pred_all = model.get_pred(adaptest_data)
        if self.I is None:
            self.I = [np.zeros((model.model.num_dim, model.model.num_dim)) for _ in range(adaptest_data.num_students)]
        selection = {}
        total_time = 0
        n = len(adaptest_data.tested[0])
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            untested_dets = []
            untested_fisher = []
            start_time = time.time()
            for qid in untested_questions:
                fisher_info = model.get_fisher(sid, qid, pred_all)
                untested_fisher.append(fisher_info)
                untested_dets.append(np.linalg.det(self.I[sid] + fisher_info))
            j = np.argmax(untested_dets)
            selection[sid] = untested_questions[j]
            self.I[sid] += untested_fisher[j]
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            print(f'Random Strategy: {sid}th student, select question: {untested_questions[j]}, cost time: {execution_time}')
        av_time = total_time / adaptest_data.num_students
        return selection, total_time, av_time

class DoptStrategy(MFIStrategy):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'D-Optimality Strategy'