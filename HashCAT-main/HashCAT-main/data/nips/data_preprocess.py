import os
import pandas as pd
import numpy as np
import json
from config import Config
from joblib import Parallel, delayed

np.random.seed(123)
pd.set_option('display.max_columns', None)

class DataProcess:
    def __init__(self, config=None):
        self.train_dataset_path = config.train_dataset_path
        self.log_json = []
        self.kc_map = {}
        self.uid_map = {}
        self.qid_map = {}
        self.knowledge_map = {}

    def read_data(self):
        raw_train_data = pd.read_csv(self.train_dataset_path)
        train_data = raw_train_data
        print('train_data success')
        self.raw_max_userid = train_data['user_id'].max()
        return {'train_data': train_data}

    def join_data(self, df_left, df_right, key):
        merge_data = pd.merge(df_left, df_right, how='left', on=key)
        return merge_data

    def encode_feat(self, df, feat_name, start_value, new_feat=None):
        all_feat_value = df[feat_name].unique().tolist()
        encode_map = dict(zip(all_feat_value, np.arange(start_value, len(all_feat_value) + start_value).tolist()))
        if new_feat:
            df[new_feat] = df[feat_name].map(encode_map)
        else:
            df[feat_name] = df[feat_name].map(encode_map)
        return df, encode_map

    def pre_process_encode(self, train_data):
        train_data, self.uid_map = self.encode_feat(train_data, 'student_id', 1)
        train_data, self.qid_map = self.encode_feat(train_data, 'question_id', 1)
        train_data, self.kc_map = self.encode_feat(train_data, 'knowledge_code', 1)
        return train_data

    def gen_train_data(self):
        all_data = self.read_data()
        feats = ['user_id', 'problem_id', 'correct', 'skill_id']
        train_data = all_data['train_data'][feats]
        train_data.loc[:, 'correct'] = (train_data['correct'] == 1).astype('int')
        train_data = train_data.rename(
            columns={'problem_id': 'question_id', 'user_id': 'student_id', 'skill_id': 'knowledge_code', 'correct': 'score'})
        old_len = len(train_data)
        train_data = train_data.dropna(axis=0, how='any')
        print('nan:', old_len - len(train_data))
        train_data = self.pre_process_encode(train_data)
        train_data.groupby('student_id').apply(self.apply_to_json)

    def save_log_map(self, path_log, path_map):
        with open(path_log, 'w', encoding='utf8') as output_file:
            print(path_log)
            json.dump(self.log_json, output_file, indent=4, ensure_ascii=False)
        with open(path_map + 'kc_map.json', 'w', encoding='utf8') as output_file:
            print(path_map + 'kc_map.json')
            json.dump(self.kc_map, output_file, indent=4, ensure_ascii=False)
        with open(path_map + 'uid_map.json', 'w', encoding='utf8') as output_file:
            print(path_map + 'uid_map.json')
            json.dump(self.uid_map, output_file, indent=4, ensure_ascii=False)
        with open(path_map + 'qid_map.json', 'w', encoding='utf8') as output_file:
            print(path_map + 'qid_map.json')
            json.dump(self.qid_map, output_file, indent=4, ensure_ascii=False)
        with open(path_map + 'knowledge_map.json', 'w', encoding='utf8') as output_file:
            print(path_map + 'knowledge_map.json')
            json.dump(self.knowledge_map, output_file, indent=4, ensure_ascii=False)

    def apply_split_record(self, df_g):
        split_cnt = 0
        user_id = df_g.iloc[0]['student_id']
        df_g['interval'] = pd.Timedelta(0)
        last_idx = -1
        last_idxs = []
        for idx, r in df_g.iterrows():
            if last_idx == -1:
                delta = pd.Timedelta(0)
            if delta > pd.Timedelta('30 days'):
                split_cnt += 1
                df_g.loc[last_idxs, 'student_id'] = '%s_%s' % (user_id, split_cnt)
                last_idxs = []
            last_idxs.append(idx)
            df_g.loc[idx, 'interval'] = delta
            last_idx = idx
        if len(last_idxs) != 0:
            split_cnt += 1
            df_g.loc[last_idxs, 'student_id'] = '%s_%s' % (user_id, split_cnt)
        return df_g

    def apply_to_json(self, df_g):
        user_id = df_g.iloc[0]['student_id']
        log_num = len(df_g)
        one_user = {'student_id': int(user_id), 'log_num': log_num, 'logs': []}
        feats = ['question_id', 'score', 'knowledge_code']
        df_g = df_g[feats]
        df_list = df_g.values.tolist()
        user_log = []
        for i in range(log_num):
            log = {
                "question_id": df_list[i][0],
                "score": df_list[i][1],
                "knowledge_code": [df_list[i][2]]
            }
            user_log.append(log)
            if log['question_id'] not in self.knowledge_map:
                self.knowledge_map[log['question_id']] = log['knowledge_code']
        one_user['logs'] = user_log
        self.log_json.append(one_user)
        return None

    def applyParallel(self, dfGrouped, func):
        retLst = Parallel(n_jobs=8)(delayed(func)(group) for name, group in dfGrouped)
        return pd.concat(retLst)

if __name__ == "__main__":
    conf = Config()
    loader = DataProcess(conf)
    loader.gen_train_data()
    loader.save_log_map(conf.new_data_save_path, conf.map_path)