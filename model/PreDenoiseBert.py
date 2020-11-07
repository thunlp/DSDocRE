import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM
import random
from tools.accuracy_tool import *

class PreDenoiseBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PreDenoiseBert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_hidden = self.bert.config.hidden_size
        self.hidden = 256

        self.linear = nn.Bilinear(self.bert_hidden, self.bert_hidden, self.hidden)
        self.rank_score = nn.Linear(self.hidden, 1)

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward_test(self, data, config):
        batch = data['doc'].shape[0]
        ent_num, men_num = data['entPos'].shape[1], data['entPos'].shape[2]
        hidden, _ = self.bert(data['doc'], encoder_attention_mask = data['attMask'])
        indice = torch.arange(0, batch).view(batch, 1, 1).repeat(1, ent_num, men_num)
        ent = hidden[indice, data['entPos']]
        ent = torch.max(ent, dim = 2)[0]

        pair_num = data['heads'].shape[1]
        indice = torch.arange(0, batch).view(batch, 1).repeat(1, pair_num)
        head = ent[indice, data['heads']] # batch, pair_num, self.bert_hidden
        tail = ent[indice, data['tails']]
        feature = self.linear(head, tail)
        score = self.rank_score(feature).squeeze(2) # batch, pair_num
        
        return {'score': score, 'loss': 0, 'titles': data['titles']}

    def forward(self, data, config, gpu_list, acc_result, mode):
        if mode == 'test':
            return self.forward_test(data, config)
        if acc_result is None:
            acc_result = {'RelDet': {'right': 0, 'all': 0}}

        batch = data['doc'].shape[0]
        ent_num, men_num = data['entPos'].shape[1], data['entPos'].shape[2]

        hidden, _ = self.bert(data['doc'], encoder_attention_mask = data['attMask'])
        indice = torch.arange(0, batch).view(batch, 1, 1).repeat(1, ent_num, men_num)
        ent = hidden[indice, data['entPos']]# batch, ent_num, men_num, self.bert_hidden

        ent = torch.max(ent, dim = 2)[0]
        rank_num = data['rankh'].shape[1]
        cand_num = data['rankh'].shape[2]
        indice = torch.arange(0, batch).view(batch, 1, 1).repeat(1, rank_num, cand_num)
        #from IPython import embed; embed()
        rdhead = ent[indice, data['rankh']].view(batch, rank_num, cand_num, self.bert_hidden)
        rdtail = ent[indice, data['rankt']].view(batch, rank_num, cand_num, self.bert_hidden)

        feature = self.linear(rdhead, rdtail) # batch, rank_num, cand_num, rel_num
        score = self.rank_score(feature).squeeze(3).view(batch * rank_num, cand_num) # batch, rank_num, cand_num
        rdlabel = data['ranklabel'].view(batch * rank_num)
        
        loss = self.loss(score, rdlabel)
        acc_result['RelDet'] = softmax_accu(score, rdlabel, acc_result['RelDet'])
        
        return {"loss": loss, 'acc_result': acc_result}
