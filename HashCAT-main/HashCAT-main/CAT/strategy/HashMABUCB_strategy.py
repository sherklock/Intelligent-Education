import os
import random
import datetime
import numpy as np
import json
import torch
import torch.nn.functional as F
import time
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset
from utils import CommonArgParser
from utils import dataset, device

args = CommonArgParser().parse_args()

question_binary_bank_path = './CAT/strategy/QuestionBinaryBank/%s_question_binary4.json' % dataset

theta = 0.001
tbeta = 8
S = 22

class HashMABUCBStrategy(AbstractStrategy):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "HashMABUCB Select Stratepy"

    def binary_gumbel_softmax(self, logits, tau=1000):
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        logits = logits.unsqueeze(-1)
        logits = torch.cat([logits, torch.zeros_like(logits)], dim=-1)
        gumbel_out = F.gumbel_softmax(logits, tau=tau, hard=True)
        return gumbel_out[..., 0]

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset, it):
        epsilon = 0.9
        selection = {}
        total_time = 0
        test_num_students = len(adaptest_data.student_ids)

        with open(question_binary_bank_path, 'r') as question:
            question_binary_bank = json.load(question)

        question_ids = np.array(list(question_binary_bank.keys()))
        question_binaries = np.stack([np.array(q['binary_code']) for q in question_binary_bank.values()])
        discriminations = np.array([q['discrimination'][0] for q in question_binary_bank.values()])
        difficulties = np.array([q['difficulty'] for q in question_binary_bank.values()])

        rewards = np.zeros(len(question_ids))
        counts = np.zeros(len(question_ids))
        min_value = 1e-5

        for sid in adaptest_data.student_ids:
            untested_questions = np.array(list(adaptest_data.untested[sid]))

            if len(untested_questions) == 0:
                continue

            sidtensor = torch.tensor(sid, dtype=torch.long)

            if model.name in ['Neural Cognitive Diagnosis', 'Multidimensional Item Response Theory']:
                student_emb = model.get_knowledge_status(sidtensor)
                student_emb_binary = self.binary_gumbel_softmax(student_emb).flatten().tolist()
                student_emb_mean = torch.mean(student_emb).item()
                student_emb_binary = np.array(student_emb_binary, dtype=np.int32)
                student_emb_binary = student_emb_binary[np.newaxis, :]
            if model.name == 'Hash Code Pre-Train':
                student_emb, student_emb_binary = model.get_knowledge_status(sidtensor)
                student_emb_binary = student_emb_binary.flatten().tolist()
                student_emb_mean = torch.mean(student_emb).item()
                student_emb_binary = np.array(student_emb_binary, dtype=np.int32)
                student_emb_binary = student_emb_binary[np.newaxis, :]

            start_time = time.time()
            untested_mask = np.isin(question_ids, untested_questions)
            filtered_question_binaries = question_binaries[untested_mask].astype(np.int32)
            filtered_discriminations = discriminations[untested_mask]
            filtered_difficulties = difficulties[untested_mask]
            filtered_question_ids = question_ids[untested_mask]

            if len(filtered_discriminations) == 0 or len(filtered_difficulties) == 0:
                continue
            if len(filtered_difficulties) > 0:
                std_difficulites = np.std(filtered_difficulties)
                if std_difficulites == 0:
                    std_difficulites = 1e-9
            else:
                continue

            hamming_distance = np.mean(student_emb_binary ^ filtered_question_binaries, axis=1)
            matching = 1 - hamming_distance
            P_correct = 1 / (1 + np.exp(-theta * matching * filtered_discriminations + tbeta * filtered_difficulties))
            H_prior = -P_correct * np.log(P_correct + 1e-5) - (1 - P_correct) * np.log(1 - P_correct + 1e-5)
            difficulty_gap = np.abs(student_emb_mean - filtered_difficulties)
            H_post = H_prior * np.exp(-difficulty_gap / (np.exp(filtered_difficulties)))
            IG = H_prior - H_post

            if it <= S:
                epsilon = max(0.01, epsilon * 0.95)
                if random.random() < epsilon:
                    best_question_id = random.choice(filtered_question_ids)
                else:
                    best_question_idx = np.argmax(IG)
                    best_question_id = str(filtered_question_ids[best_question_idx])
            else:
                filtered_rewards = rewards[np.isin(question_ids, filtered_question_ids)]
                filtered_counts = counts[np.isin(question_ids, filtered_question_ids)]

                total_counts = np.sum(filtered_counts)
                ucb_values = filtered_rewards / (filtered_counts + 1e-9) + np.sqrt(
                    2 * np.log(total_counts + 1) / (filtered_counts + 1e-9))
                best_question_idx = np.argmax(ucb_values)
                best_question_id = str(filtered_question_ids[best_question_idx])

            selected_idx = np.where(filtered_question_ids == best_question_id)[0][0]

            original_idx = np.where(question_ids == filtered_question_ids[selected_idx])[0][0]
            rewards[original_idx] += IG[selected_idx]
            counts[original_idx] += 1

            selection[sid] = int(best_question_id)
            end_time = time.time()

            execution_time = end_time - start_time

            total_time += execution_time

        avg_time = total_time / test_num_students
        return selection, total_time, avg_time