import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset
import random
import time
class BECATStrategy(AbstractStrategy):
    
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'BECAT Strategy'
    
    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset,S_set):
        assert hasattr(model, 'delta_q_S_t'), \
            'the models must implement delta_q_S_t method'
        assert hasattr(model, 'get_pred'), \
            'the models must implement get_pred method for accelerating'
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        total_time = 0
        test_num_students = len(adaptest_data.student_ids)
        for sid in adaptest_data.student_ids:
            tmplen = (len(S_set[sid]))
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            start_time = time.time()
            sampled_elements = np.random.choice(untested_questions, tmplen + 5)
            untested_deltaq = [model.delta_q_S_t(qid, pred_all[sid],S_set[sid],sampled_elements) for qid in untested_questions]
            j = np.argmax(untested_deltaq)
            selection[sid] = untested_questions[j]
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            print(f'BECAT Strategy: {sid}th student, select question: {j}, cost time: {execution_time}')
        avg_time = total_time / test_num_students
        return selection, total_time, avg_time
    