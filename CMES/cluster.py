import json
from sklearn.cluster import AgglomerativeClustering
import numpy as np

n_clusters=20

with open('data/train_set.json', 'r') as file:
    data = json.load(file)

user_ids = [item['user_id']-1 for item in data]
exer_ids = [item['exer_id']-1 for item in data]
scores = [item['score'] for item in data]
knowledge_codes = [item['knowledge_code'] for item in data]

all_knowledge_codes = set()
for item in data:
    knowledge = item['knowledge_code']
    all_knowledge_codes.update(knowledge)

user_exer_matrix = {}
for user, exer, score, codes in zip(user_ids, exer_ids, scores, knowledge_codes):
    if user not in user_exer_matrix:
        user_exer_matrix[user] = {}
    # user_exer_matrix[user][exer] = {'score': score, 'codes': codes}
    for code in codes:
        user_exer_matrix[user][code] = {'score': score}

user_ids = list(set([item['user_id'] - 1 for item in data]))
user_knowledge_matrix = {}
for user in user_ids:
    if user not in user_knowledge_matrix:
        user_knowledge_matrix[user] = []

for item in data:
    user = item['user_id']-1
    knowledge = item['knowledge_code']
    user_knowledge_matrix[user].extend(knowledge)

# user_knowledge_list = []
# for user, knowledge in user_knowledge_matrix.items():
#     feature_vector = [1 if code in knowledge else 0 for code in all_knowledge_codes]
#     user_knowledge_list.append(feature_vector)

user_knowledge_list = []
for user, knowledge in user_knowledge_matrix.items():
    feature_vector = []
    for code in all_knowledge_codes:
        if code in knowledge:
            if user_exer_matrix[user][code]['score'] == 1:
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        else:
            feature_vector.append(-1)
    user_knowledge_list.append(feature_vector)

clustering = AgglomerativeClustering(n_clusters)
user_cluster_labels = clustering.fit_predict(user_knowledge_list)

clusters = {}
for i, label in enumerate(user_cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(user_ids[i])

new_clusters = {}
for label, users in clusters.items():
    new_clusters[str(label)] = users

save_name='result'+str(n_clusters)+'.json'
with open(save_name, 'w') as file:
    json.dump(new_clusters, file)

print("Clustering completed and result saved to result.json.")
