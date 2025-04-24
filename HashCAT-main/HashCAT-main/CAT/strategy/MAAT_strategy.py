import numpy as np
import torch
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset
import time

class MAATStrategy(AbstractStrategy):
    def __init__(self, n_candidates=10):
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'Model Agnostic Adaptive Testing'

    def _compute_coverage_gain(self, sid, qid, adaptest_data: AdapTestDataset):
        concept_cnt = {}
        for q in adaptest_data.data[sid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] = 0
        for q in list(adaptest_data.tested[sid]) + [qid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] += 1
        return (sum(cnt / (cnt + 1) for c, cnt in concept_cnt.items())
                / sum(1 for c in concept_cnt))

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'expected_model_change'), \
            'the models must implement expected_model_change method'
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        total_time = 0
        test_num_students = len(adaptest_data.student_ids)
        print('test number_students:', test_num_students)
        for sid in adaptest_data.student_ids:
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            start_time = time.time()
            emc_arr = [model.expected_model_change(sid, qid, adaptest_data, pred_all) for qid in untested_questions]
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            qidx = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data))
            selection[sid] = qidx
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            print(f'Model Agnostic Adaptive Testing: {sid}th student, select question: {qidx}, cost time: {execution_time}')
        avg_time = total_time / test_num_students
        return selection, total_time, avg_time

    def sel_fisher(self, model, adaptest_data: AdapTestDataset):
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = torch.tensor(list(adaptest_data.untested[sid]))
            theta = np.array(model.model.theta(torch.tensor(sid)).detach().numpy())
            alpha = np.array(model.model.alpha(untested_questions).detach().numpy())
            beta = np.array(model.model.beta(untested_questions).detach().numpy())
            fisher = self.fisher_information(model, alpha, beta, theta)
            selection[sid] = untested_questions[np.argmax(fisher)].item()
        return selection

    def fisher_information(self, model, alpha, beta, theta):
        try:
            information = []
            for t in theta:
                p = model.irf(alpha, beta, t)
                q = 1 - p
                information.append(p * q * (alpha ** 2))
            information = np.array(information)
            return information
        except TypeError:
            p = model.irf(alpha, beta, theta)
            q = 1 - p
            return (p * q * (alpha ** 2))