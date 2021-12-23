# -*- coding: utf-8 -*-

"""
关系分类
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertRelCls(nn.Module):
    def __init__(self, hidden_size=768, class_num=3):
        super(BertRelCls, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(3*hidden_size, class_num)
        self.bert = BertModel.from_pretrained("bert-base-chinese")

    def forward(self, input_ids, token_type_ids, attention_mask, entity_1_mask, entity_2_mask):
        """
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param entity_1_mask:
                batch_size*seq_len
                [[0, 1, 1, 1, 0], [1, 1, 0, 0, 0]] 1表示实体1所在位置
        :param entity_2_mask:
                batch_size*seq_len
                [[0, 0, 1, 1, 0], [1, 1, 1, 1, 0]] 1表示实体2所在位置
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids, attention_mask)

        # cls
        cls = self.fc1(F.tanh(outputs[1]))

        # 所有字向量
        hiddens = outputs[0]  # batch_size * seq_len * hidden_size

        # 实体长度
        entity_1_len = torch.sum(entity_1_mask, dim=1, keepdim=True)
        entity_2_len = torch.sum(entity_2_mask, dim=1, keepdim=True)

        # 新增一个维度
        entity_1_mask = entity_1_mask.unsqueeze(-1)
        entity_2_mask = entity_2_mask.unsqueeze(-1)

        # 将实体所在向量求均值
        entity_1_mean = torch.sum(hiddens * entity_1_mask, dim=1) / entity_1_len
        entity_2_mean = torch.sum(hiddens * entity_2_mask, dim=2) / entity_2_len

        # 实体均值向量，激活+Fc
        entity_1_mean = self.fc2(F.tanh(entity_1_mean))
        entity_2_mean = self.fc2(F.tanh(entity_2_mean))

        # 将[cls] + [entity1] + [entity2] 拼接起来
        final = torch.cat([cls, entity_1_mean, entity_2_mean], dim=1)

        # 分类
        out = self.fc3(final)

