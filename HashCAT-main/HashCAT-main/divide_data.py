import json
import random
import math
import numpy as np

min_log = 10
min_exer = 50
min_kc = 10
min_cat_log = 50

def preprocess(data_name):
    with open('data/%s/log_data.json' % data_name, encoding='utf8') as i_f:
        stus = json.load(i_f)
    filter_dump_stus = []
    for stu in stus:
        exer_cnt = {}
        for log in stu['logs']:
            if log["exer_id"] not in exer_cnt:
                exer_cnt[log["exer_id"]] = {'cnt': 0, 1: 0, 0: 0}
            exer_cnt[log["exer_id"]]['cnt'] += 1
            exer_cnt[log["exer_id"]][log['score']] += 1
        new_logs = []
        for log in stu['logs']:
            if log["exer_id"] in exer_cnt:
                if exer_cnt[log["exer_id"]][1] > exer_cnt[log["exer_id"]][0]:
                    log["score"] = 1
                else:
                    log["score"] = 0
                new_logs.append(log)
                del exer_cnt[log["exer_id"]]
        filter_dump_stus.append({"user_id": stu["user_id"], "log_num": len(new_logs), "logs": new_logs})
    stus = filter_dump_stus

    exer_stu = {}
    kc_exer = {}
    for stu in stus:
        for log in stu['logs']:
            if log["exer_id"] not in exer_stu:
                exer_stu[log["exer_id"]] = set()
            exer_stu[log["exer_id"]].add(stu['user_id'])
            for kc in log['knowledge_code']:
                if kc not in kc_exer:
                    kc_exer[kc] = set()
                kc_exer[kc].add(log["exer_id"])

    filter_stus = []
    cnt = 0
    exer_set = set()
    for stu in stus:
        logs = []
        for log in stu['logs']:
            if len(exer_stu[log["exer_id"]]) >= min_exer:
                new_kc = []
                for kc in log['knowledge_code']:
                    if len(kc_exer[kc]) >= min_kc:
                        new_kc.append(kc)
                if len(new_kc) == 0:
                    continue
                log['knowledge_code'] = new_kc
                logs.append(log)
            else:
                cnt += 1
        if len(set([log['exer_id'] for log in logs])) >= min_log:
            filter_stus.append({"user_id": stu["user_id"], "log_num": len(logs), "logs": logs})
            exer_set = exer_set.union(set([log['exer_id'] for log in logs]))

    item_id_map = dict(zip(exer_set, np.arange(1, len(exer_set) + 1).tolist()))
    new_stus = []
    log_all, exer_set, kc_set = 0, set(), set()
    for stu in filter_stus:
        logs = []
        for log in stu['logs']:
            log['exer_id'] = item_id_map[log['exer_id']]
            logs.append(log)
        new_stus.append({"user_id": stu["user_id"], "log_num": len(logs), "logs": logs})
        log_all += len(set([log['exer_id'] for log in logs]))
        exer_set = exer_set.union(set([log['exer_id'] for log in logs]))
        for log in logs:
            kc_set = kc_set.union(set(log['knowledge_code']))

    print(cnt)
    stu_all = len(new_stus)
    exer_all = len(exer_set)
    exer_maxid = max(exer_set)
    stu_maxid = max(set([s['user_id'] for s in stus]))
    kc_maxid = max(kc_set)
    kc_all = len(kc_set)
    filtered_info = f"data_name: {data_name}\nstu_all: {stu_all}\nstu_maxid: {stu_maxid}\nexer_all: {exer_all}\nexer_maxid: {exer_maxid}\nkc_all: {kc_all}\nkc_maxid: {kc_maxid}\nlog_all: {log_all}\navg_per_stu: {log_all/stu_all}\n"
    with open(f'data/{data_name}/info_filtered.yml', 'w', encoding='utf8') as output_file:
        print(filtered_info)
        output_file.write(filtered_info)
    with open(f'data/{data_name}/log_data_filtered.json', 'w', encoding='utf8') as output_file:
        json.dump(new_stus, output_file, indent=4, ensure_ascii=False)

def divide_data(data_name, min_cat_log=5):
    with open(f'data/{data_name}/log_data_filtered_encoded.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    stus = [stu for stu in stus if len(stu['logs']) >= min_cat_log]
    stu_len = len(stus)
    print(f'all stu len: {stu_len}')
    train_size = int(stu_len * 0.7)
    val_size = int(stu_len * 0.1)
    train_set = stus[:train_size]
    val_set = stus[train_size:train_size + val_size]
    test_set = stus[train_size + val_size:]
    print(f'train_set: {len(train_set)}, val_set: {len(val_set)}, test_set: {len(test_set)}')
    with open(f'data/{data_name}/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open(f'data/{data_name}/val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)
    with open(f'data/{data_name}/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)

def divide_data2(data_name, min_cat_log=5):
    with open(f'data/{data_name}/log_data_filtered_encoded.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    stus = [stu for stu in stus if len(stu['logs']) >= min_cat_log]
    random.shuffle(stus)
    stu_len = len(stus)
    train_stu_size = math.ceil(stu_len * 0.7)
    val_stu_size = math.ceil(stu_len * 0.1)
    test_stu_size = stu_len - train_stu_size - val_stu_size
    train_students = stus[:train_stu_size]
    val_students = stus[train_stu_size:train_stu_size + val_stu_size]
    test_students = stus[train_stu_size + val_stu_size:]
    train_set = []
    val_set = []
    test_set = []
    for stu in train_students:
        logs = stu['logs']
        log_len = len(logs)
        train_size = math.ceil(log_len * 0.7)
        val_size = math.ceil(log_len * 0.1)
        stu_train_logs = logs[:train_size]
        stu_val_logs = logs[train_size:train_size + val_size]
        stu_test_logs = logs[train_size + val_size:]
        if stu_train_logs:
            train_set.append({'student_id': stu['student_id'], 'logs': stu_train_logs})
        if stu_val_logs:
            val_set.append({'student_id': stu['student_id'], 'logs': stu_val_logs})
        if stu_test_logs:
            test_set.append({'student_id': stu['student_id'], 'logs': stu_test_logs})
    for stu in val_students:
        logs = stu['logs']
        log_len = len(logs)
        train_size = math.ceil(log_len * 0.7)
        val_size = math.ceil(log_len * 0.1)
        stu_train_logs = logs[:train_size]
        stu_val_logs = logs[train_size:train_size + val_size]
        stu_test_logs = logs[train_size + val_size:]
        if stu_train_logs:
            train_set.append({'student_id': stu['student_id'], 'logs': stu_train_logs})
        if stu_val_logs:
            val_set.append({'student_id': stu['student_id'], 'logs': stu_val_logs})
        if stu_test_logs:
            test_set.append({'student_id': stu['student_id'], 'logs': stu_test_logs})
    for stu in test_students:
        logs = stu['logs']
        log_len = len(logs)
        train_size = math.ceil(log_len * 0.7)
        val_size = math.ceil(log_len * 0.1)
        stu_train_logs = logs[:train_size]
        stu_val_logs = logs[train_size:train_size + val_size]
        stu_test_logs = logs[train_size + val_size:]
        if stu_train_logs:
            train_set.append({'student_id': stu['student_id'], 'logs': stu_train_logs})
        if stu_val_logs:
            val_set.append({'student_id': stu['student_id'], 'logs': stu_val_logs})
        if stu_test_logs:
            test_set.append({'student_id': stu['student_id'], 'logs': stu_test_logs})
    print(f'train_set: {len(train_set)}, val_set: {len(val_set)}, test_set: {len(test_set)}')
    with open(f'data/{data_name}/train_set2.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open(f'data/{data_name}/val_set2.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)
    with open(f'data/{data_name}/test_set2.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)

def divide_data_by_logs(data_name, min_cat_log=5):
    with open(f'data/{data_name}/log_data_filtered_encoded.json', encoding='utf8') as f:
        stus = json.load(f)
    stus = [stu for stu in stus if len(stu['logs']) >= min_cat_log]
    train_set = []
    val_set = []
    test_set = []
    for stu in stus:
        logs = stu['logs']
        log_len = len(logs)
        train_size = math.ceil(log_len * 0.7)
        val_size = math.ceil(log_len * 0.1)
        test_size = log_len - train_size - val_size
        stu_train_logs = logs[:train_size]
        stu_val_logs = logs[train_size:train_size + val_size]
        stu_test_logs = logs[train_size + val_size:]
        if stu_train_logs:
            train_set.append({'student_id': stu['student_id'], 'logs': stu_train_logs})
        if stu_val_logs:
            val_set.append({'student_id': stu['student_id'], 'logs': stu_val_logs})
        if stu_test_logs:
            test_set.append({'student_id': stu['student_id'], 'logs': stu_test_logs})
    print(f'train_set: {len(train_set)}, val_set: {len(val_set)}, test_set: {len(test_set)}')
    with open(f'data/{data_name}/train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open(f'data/{data_name}/val_set.json', 'w', encoding='utf8') as output_file:
        json.dump(val_set, output_file, indent=4, ensure_ascii=False)
    with open(f'data/{data_name}/test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    dataset_name = 'nips'
    print(dataset_name)
    divide_data2(dataset_name)