import datetime
import json
import logging
import os
import random

import numpy as np

logging.basicConfig(level=logging.INFO)


class AdapTestDriver(object):

    @staticmethod
    def run(model, strategy, adaptest_data, test_length, log_dir, datasetname):
        result_dir = os.path.join(log_dir, f"{strategy.name}_{model.name}_{datasetname}")
        os.makedirs(result_dir, exist_ok=True)
        log_file_path = os.path.join(result_dir, 'results_log.txt')
        print(f'start adaptive testing with {strategy.name} ')
        print(f'using dataset is {datasetname}\n')
        print(f'dataset test_num_students is {len(adaptest_data.student_ids)}\n')
        print(f'using CDM is {model.name}\n')
        print(f'Interation 0')

        with open(log_file_path, 'a') as log_file:
            log_file.write(f'start adaptive testing with {strategy.name} strategy\n')
            log_file.write(f'using dataset is {datasetname}\n')
            log_file.write(f'dataset test_num_students is {len(adaptest_data.student_ids)}\n')
            log_file.write(f'using CDM is {model.name}\n')
            log_file.write(f'Interation 0\n')

        results = model.evaluate(adaptest_data)

        for name, value in results.items():
            print(f'{name}:{value}')
        with open(log_file_path, 'a') as log_file:
            for name, value in results.items():
                log_file.write(f'{name}:{value}\n')

        S_sel = {}
        for sid in range(adaptest_data.num_students):
            key = sid
            S_sel[key] = []
        selected_questions = {}

        if strategy.name == 'BOBCAT':
            real = {}
            real_data = adaptest_data.data
            for sid in real_data:
                question_ids = list(real_data[sid].keys())
                real[sid] = {}
                tmp = {}
                for qid in question_ids:
                    tmp[qid] = real_data[sid][qid]
                real[sid] = tmp
        student_choice_logs = {}
        time = 0
        avg_time = 0
        total_time = 0
        total_avg_time = 0

        for it in range(1, test_length + 1):
            print(f'Iteration {it}')
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'Iteration {it}\n')

            if strategy.name == 'BOBCAT':
                selected_questions, time, avg_time = strategy.adaptest_select(model, adaptest_data, S_sel)
                for sid in adaptest_data.student_ids:
                    tmp = {}
                    tmp[selected_questions[sid]] = real[sid][selected_questions[sid]]
                    S_sel[sid].append(tmp)
            if it == 1 and strategy.name == 'BECAT Strategy':
                for sid in adaptest_data.student_ids:
                    untested_questions = np.array(list(adaptest_data.untested[sid]))
                    random_index = random.randint(0, len(untested_questions) - 1)
                    selected_questions[sid] = untested_questions[random_index]
                    S_sel[sid].append(untested_questions[random_index])
            elif strategy.name == 'BECAT Strategy':
                selected_questions, time, avg_time = strategy.adaptest_select(model, adaptest_data, S_sel)
                for sid in adaptest_data.student_ids:
                    S_sel[sid].append(selected_questions[sid])
            # time: 这一轮所有学生选题的时间
            elif strategy.name == 'HashMABUCB Select Stratepy':

                selected_questions, time, avg_time = strategy.adaptest_select(model,
                                                                              adaptest_data,
                                                                              it)

            else:
                selected_questions, time, avg_time = strategy.adaptest_select(model, adaptest_data)

            total_time += time
            total_avg_time += avg_time

            for student, question in selected_questions.items():
                adaptest_data.apply_selection(student, question)

            model.adaptest_update(adaptest_data)

            results = model.evaluate(adaptest_data)

            with open(log_file_path, 'a') as log_file:
                if it % 5 == 0:
                    log_file.write(f'Total Time: {total_time}\n')
                    log_file.write(f'Avg Time: {total_avg_time}\n')
                for name, value in results.items():
                    log_file.write(f'{name}:{value}\n')

            # # log results
            for name, value in results.items():
                print(f'{name}:{value}')
