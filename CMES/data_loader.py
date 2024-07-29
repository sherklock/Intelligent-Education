import json
import torch
from collections import Counter
import csv
import random

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self,topk,batch_size):
        self.topk = topk
        n_cluster = 50
        save_name = 'result' + str(n_cluster) + '.json'
        with open(save_name, 'r') as f:
            self.clustering_results = json.load(f)
        self.batch_size = batch_size
        self.ptr = 0
        self.data = []

        data_file = 'assist/train_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.exer_knowledge_data, self.kn_max_length = self.load_data_from_csv('item.csv')
        self.user_interactions = self.get_user_interactions()

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys, neg_pos_exer_ids = [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)
            
            matching_knowledge_code = self.exer_knowledge_data.get(log['exer_id'] - 1)

            # if y:
            result = self.get_matching_exer(self.user_interactions, log['user_id'] - 1)
            top_exer_ids = self.get_same_kn_exer_ids(result, matching_knowledge_code, log['exer_id'] - 1)
            neg_pos_exer_ids.append(top_exer_ids)
            # else:
            #     result = self.get_matching_exer(self.user_interactions_neg, log['user_id'] - 1)
            #     top_exer_ids = self.get_same_kn_exer_ids(result, matching_knowledge_code, log['exer_id'] - 1)
            #     neg_pos_exer_ids.append(top_exer_ids)
        
        # print("neg_pos_exer_ids",neg_pos_exer_ids)
        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys), torch.LongTensor(neg_pos_exer_ids)

    def load_data_from_csv(self,file_path):
        data = {}
        
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  
            
            for row in reader:
                item_id = int(row[0])
                knowledge_code = eval(row[1])
                
                if item_id not in data:
                    data[item_id] = []
                
                data[item_id].extend([x - 1 for x in knowledge_code])
        data = {item_id: list(set(lst)) for item_id, lst in data.items()}
        max_length = max(len(lst) for lst in data.values())

        return data,max_length

    def get_key_from_value(self,dictionary, target_value):
        for key, value in dictionary.items():
            if target_value in value:
                return key
        return None

    def get_other_users(self,cluster_data, cur_cluster):
        other_users = []

        for cluster, users in cluster_data.items():
            if cluster != cur_cluster:
                other_users.extend(users)
                break
        
        return other_users

    def get_user_interactions(self):
        user_interactions = {}

        for log in self.data:
            user_id = log['user_id']-1
            exer_id = log['exer_id']-1
            if user_id in user_interactions:
                user_interactions[user_id].append(exer_id)
            else:
                user_interactions[user_id] = [exer_id]

        return user_interactions

    def get_matching_exer(self, dict_B, user_id):
        matching_students = []

        # matching_knowledge_code = self.exer_knowledge_data.get(log['exer_id'] - 1)
        cluster=self.get_key_from_value(self.clustering_results,user_id)

        matching_students=self.get_other_users(self.clustering_results,cluster)

        matching_exer_ids = []

        for stu_id in matching_students:
            if stu_id in dict_B:
                matching_exer_ids.extend(dict_B[stu_id])
        return matching_exer_ids
        
    def get_same_kn_exer_ids(self, matching_exer_ids, choose_exer_kn,cur_exer_id):
        matching_questions = []
        for exer_id in matching_exer_ids:
            if exer_id != cur_exer_id:
            # if exer_id not in self.exer_knowledge_data:
                knowledge_codes = self.exer_knowledge_data.get(exer_id)
                if knowledge_codes == choose_exer_kn:
                    matching_questions.append(exer_id)
                # elif knowledge_codes in choose_exer_kn:
                #     matching_questions.append(exer_id)

        if len(matching_questions)<self.topk:
            for i in range(len(matching_questions)):
                matching_questions.append(matching_questions[i])
            if len(matching_questions)<self.topk:
                for _ in range(self.topk-len(matching_questions)):
                    matching_questions.append(cur_exer_id)

        matching_counts = Counter(matching_questions)

        matching_questions.sort(key=lambda x: matching_counts[x], reverse=True)

        top_questions = matching_questions[:self.topk]

        return top_questions

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

class ValTestDataLoader(object):
    def __init__(self, d_type='validation'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if d_type == 'validation':
            data_file = 'assist/val_set.json'
        else:
            data_file = 'assist/test_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            ys.append(y)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
