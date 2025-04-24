import numpy as np
import time
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset

class RandomStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Random Select Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        selection = {}
        total_time = 0
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))

            start_time = time.time()
            random_question_id = np.random.randint(len(untested_questions))

            qidx = untested_questions[random_question_id]
            selection[sid] = qidx
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            print(f'Random Strategy: {sid}th student, select question: {qidx}, cost time: {execution_time}')
        av_time = total_time / adaptest_data.num_students
        return selection, total_time, av_time