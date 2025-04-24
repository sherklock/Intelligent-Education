import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import vegas
from sklearn.metrics import roc_auc_score, accuracy_score
import math
from CAT.model.abstract_model import AbstractModel
from scipy import integrate
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset
from utils import device
from torch.utils.data import DataLoader


# device = torch.device('cpu')
class NCD(nn.Module):
    '''
    NeuralCDM
    '''

    def __init__(self, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64, load_path=None):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2  # changeable

        super(NCD, self).__init__()

        # network structure
        # 使用学生数量和知识点数量对学生做embedding
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim).to(device)
        # 使用题目数量和知识点数量为题目难度做embedding
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim).to(device)
        # 区分度
        self.e_discrimination = nn.Embedding(self.exer_n, 1).to(device)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1).to(device)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2).to(device)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1).to(device)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        # print(self.student_emb.weight)

        if load_path is not None:
            self.load_model(load_path)
            for name, param in self.named_parameters():
                if 'student_emb' not in name:
                    param.requires_grad = False

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def load_model(self, path):
        state_dict = torch.load(path)
        load_dict = {k: v for k, v in state_dict.items() if 'student' not in k}
        self.load_state_dict(load_dict, strict=False)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


def monte_carlo_integral(kli, boundaries, samples=10000, device=device):
    dimensions = len(boundaries)

    # 在 GPU 上生成随机采样点
    samples_tensor = torch.rand((samples, dimensions), device=device)
    for i, (lower, upper) in enumerate(boundaries):
        samples_tensor[:, i] = lower + (upper - lower) * samples_tensor[:, i]

    # 在采样点上评估被积函数
    values = torch.stack([torch.tensor(kli(sample.cpu().numpy())) for sample in samples_tensor], dim=0).to(device)

    # 计算积分结果
    volume = torch.prod(torch.tensor([upper - lower for lower, upper in boundaries], device=device))
    integral = torch.mean(values) * volume
    return integral.item()


class NCDModel(AbstractModel):

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None

    @property
    def name(self):
        return 'Neural Cognitive Diagnosis'

    def init_model(self, data: Dataset, load_path=None):
        self.model = NCD(data.num_students, data.num_questions, data.num_concepts, self.config['prednet_len1'],
                         self.config['prednet_len2'], load_path)

    def train(self, train_data: TrainDataset, valid_data: TrainDataset):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        self.model.to(device)
        logging.info(self.name + "train on {}".format(device))
        avg_losses = []
        aucs = []
        accs = []
        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            loss = []
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                pred = pred.to(device)
                # print("pred shape:", pred.shape)
                # print("labels shape:", labels.shape)
                # print("pred values:", pred)
                # print("labels values:", labels)

                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss.append(bz_loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (ep, float(np.mean(loss))))
            #     total_loss += bz_loss.item()  # 使用 item() 获取标量值
            #     if cnt % log_step == 0:
            #         avg_loss = total_loss / (cnt + 1)  # 计算当前批次的平均loss
            #         logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, avg_loss))
            #
            # # 在每个epoch结束时可以打印一次平均loss
            # avg_loss_epoch = total_loss / len(train_loader)
            # logging.info('Epoch [{}]: average loss={:.5f}'.format(ep, avg_loss_epoch))
            avg_loss, auc, acc = self.validata(valid_data)
            print("Validation Loss: %.6f, AUC: %.4f, Accuracy: %.4f" % (avg_loss, auc, acc))
            avg_losses.append(avg_loss)
            aucs.append(auc)
            accs.append(acc)

    def validata(self, val_data: TrainDataset):
        device = self.config['device']
        val_loader = DataLoader(val_data, batch_size=self.config['batch_size'], shuffle=True)
        all_preds = []
        all_labels = []
        loss = []
        with torch.no_grad():  # 在验证阶段不需要计算梯度
            self.model.eval()  # 切换到评估模式
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(val_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                loss.append(bz_loss.mean().item())
                all_preds += pred.tolist()
                all_labels += labels.tolist()
            self.model.train()
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        avg_loss = np.mean(loss)
        auc = roc_auc_score(all_labels, all_preds)
        threshold = 0.5
        binary_pred = (all_preds >= threshold).astype(int)
        acc = accuracy_score(all_labels, binary_pred)

        return avg_loss, auc, acc

    def _loss_function(self, pred, real):
        pred_0 = torch.ones(pred.size()).to(self.config['device']) - pred
        output = torch.cat((pred_0, pred), dim=1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)

    def adaptest_save(self, path):
        """
        Save the model. Do not save the parameters for students.
        """
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'student' not in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        pt = torch.load(path)
        self.model.load_state_dict(pt, strict=False)
        self.model.to(self.config['device'])

    def adaptest_update(self, adaptest_data: AdapTestDataset):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.student_emb.parameters(), lr=lr)

        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_steps = 100
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device)
                concepts_emb = concepts_emb.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
                # if cnt % log_steps == 0:
                # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))
        return loss

    def evaluate(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1)
                pred += output.tolist()
            self.model.train()

        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts.update(set(concept_map[qid]))
            for qid in adaptest_data.tested[sid]:
                tested_concepts.update(set(concept_map[qid]))
            coverage = len(tested_concepts) / len(all_concepts)
            coverages.append(coverage)
        cov = sum(coverages) / len(coverages)

        real = np.array(real)
        real = np.where(real < 0.5, 0.0, 1.0)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)

        # Calculate accuracy
        threshold = 0.5  # You may adjust the threshold based on your use case
        binary_pred = (pred >= threshold).astype(int)
        acc = accuracy_score(real, binary_pred)

        return {
            'auc': auc,
            'cov': cov,
            'acc': acc
        }

    def get_knowledge_status(self, stu_id):
        stu_id = stu_id.to(device)
        return self.model.student_emb(stu_id)

    def get_exer_params(self, exer_id):
        exer_id = exer_id.to(device)
        return self.model.get_exer_params(exer_id)

    def get_pred(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    # 自定义蒙卡特罗积分

    # kli for ncd
    def get_kli(self, student_id, question_id, n, pred_all):
        if n == 0:
            return np.inf
        device = self.config['device']
        c = 3
        sid = torch.LongTensor([student_id]).to(device)
        qid = torch.LongTensor([question_id]).to(device)

        student_emb = self.model.student_emb(sid).squeeze(0)
        k_difficulty = self.model.k_difficulty(qid).squeeze(0)
        e_discrimination = self.model.e_discrimination(qid).squeeze()
        dim = student_emb.size(0)
        pred_estimate = pred_all[student_id][question_id]

        boundaries = []
        for i in range(dim):
            lower = student_emb[i].item() - c / np.sqrt(n)
            upper = student_emb[i].item() + c / np.sqrt(n)
            if lower > upper:
                lower, upper = upper, lower
            boundaries.append([lower, upper])

        total_kli = 0.0
        for i in range(dim):
            def kli_1d(x):
                # 计算预测值
                pred = k_difficulty[i].item() * x + e_discrimination.item()
                pred = 1 / (1 + np.exp(-pred))  # Logistic function

                # KL 散度计算
                q_estimate = 1 - pred_estimate
                q = 1 - pred

                # 避免 log(0) 的情况
                pred = np.clip(pred, 1e-10, 1 - 1e-10)
                q = np.clip(q, 1e-10, 1 - 1e-10)

                # 计算 KL 散度
                kl_value = pred_estimate * np.log(pred_estimate / pred) + \
                           q_estimate * np.log(q_estimate / q)
                return kl_value

            try:
                v, err = integrate.quad(kli_1d, boundaries[i][0], boundaries[i][1])
                total_kli += v
            except Exception as e:
                print(f"Integration failed for dimension {i}: {e}")
                # 根据需求决定如何处理失败的积分，例如跳过或返回一个默认值
                continue

            return total_kli

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = adaptest_data.concept_map[qid]
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        # pred = self.model(student_id, question_id, concepts_emb).item()
        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
            (1 - pred) * torch.norm(neg_weights - original_weights).item()

    def get_BE_weights(self, pred_all):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        d = 100
        Pre_true = {}
        Pre_false = {}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
        w_ij_matrix = {}
        for i, _ in pred_all.items():
            w_ij_matrix[i] = {}
            for j, _ in pred_all.items():
                w_ij_matrix[i][j] = 0
        for i, _ in pred_all.items():
            for j, _ in pred_all.items():
                criterion_true_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 1)
                criterion_false_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 1)
                criterion_true_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 0)
                criterion_false_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 0)
                tensor_11 = torch.tensor(Pre_true[i], requires_grad=True)
                tensor_12 = torch.tensor(Pre_true[j], requires_grad=True)
                loss_true_1 = criterion_true_1(tensor_11, torch.tensor(1.0))
                loss_false_1 = criterion_false_1(tensor_11, torch.tensor(0.0))
                loss_true_0 = criterion_true_0(tensor_12, torch.tensor(1.0))
                loss_false_0 = criterion_false_0(tensor_12, torch.tensor(0.0))
                loss_true_1.backward()
                grad_true_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_false_1.backward()
                grad_false_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_true_0.backward()
                grad_true_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                loss_false_0.backward()
                grad_false_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                diff_norm_00 = math.fabs(grad_true_1 - grad_true_0)
                diff_norm_01 = math.fabs(grad_true_1 - grad_false_0)
                diff_norm_10 = math.fabs(grad_false_1 - grad_true_0)
                diff_norm_11 = math.fabs(grad_false_1 - grad_false_0)
                Expect = Pre_false[i] * Pre_false[j] * diff_norm_00 + Pre_false[i] * Pre_true[j] * diff_norm_01 + \
                         Pre_true[i] * Pre_false[j] * diff_norm_10 + Pre_true[i] * Pre_true[j] * diff_norm_11
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    def F_s_func(self, S_set, w_ij_matrix):
        res = 0.0
        for w_i in w_ij_matrix:
            if (w_i not in S_set):
                mx = float('-inf')
                for j in S_set:
                    if w_ij_matrix[w_i][j] > mx:
                        mx = w_ij_matrix[w_i][j]
                res += mx

        return res

    def delta_q_S_t(self, question_id, pred_all, S_set, sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            v: float, Each weight information
        """

        Sp_set = list(S_set)
        b_array = np.array(Sp_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
        sampled_dict = {key: value for key, value in pred_all.items() if key in sampled_elements}

        w_ij_matrix = self.get_BE_weights(sampled_dict)

        F_s = self.F_s_func(Sp_set, w_ij_matrix)

        Sp_set.append(question_id)
        F_sp = self.F_s_func(Sp_set, w_ij_matrix)
        return F_sp - F_s
