import torch
from torch import nn
import torch.nn.functional as F


# NCDM
class NCDNet(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        super(NCDNet, self).__init__()

        self.sample_size = 10
        self.fusion_gate = nn.Linear(2 * self.stu_dim, self.stu_dim)
        self.proj_layer = nn.Linear(self.stu_dim, self.stu_dim)

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point, mean=None, std=None, sample=False):
        # before prednet
        stu_emb = self.student_emb(stu_id)

        if sample:
            samples_z = torch.randn((self.sample_size, mean.shape[0], mean.shape[1])).to(stu_id.device)
            samples_z = samples_z * std + mean
            samples_z_mean = samples_z.mean(dim=0)  # (B, stu_dim)
            concat_emb = torch.cat([stu_emb, samples_z_mean], dim=-1)  # (B, 2 * stu_dim)
            gate = torch.sigmoid(self.fusion_gate(concat_emb))  # [B, stu_dim]
            fusion = gate * stu_emb + (1 - gate) * samples_z_mean
        else:
            concat_emb = torch.cat([stu_emb, mean], dim=-1)  # (B, 2 * stu_dim)
            gate = torch.sigmoid(self.fusion_gate(concat_emb))  # [B, stu_dim]
            fusion = gate * stu_emb + (1 - gate) * mean

        fused_emb = self.proj_layer(fusion)

        stat_emb = torch.sigmoid(fused_emb)

        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point

        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
