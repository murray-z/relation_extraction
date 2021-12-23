# -*- coding: utf-8 -*-

"""
bert+softmax 实现序列标注
"""

import torch
import torch.nn as nn
from transformers import BertModel
from data_helper import InAnaDataset


class BertSoftmax(nn.Module):
    def __init__(self, hidden_size, class_num, drop_ratio=0.1):
        super(BertSoftmax, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.fc = nn.Linear(hidden_size, class_num)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        hidden = self.dropout(outputs[0])
        out = self.fc(hidden)
        return out


