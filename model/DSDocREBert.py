import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM
import random
from tools.accuracy_tool import *
# rank: Relation Detection
# rel: Relational Fact Alignment
# men: Mention-Entity Matching

class DSDocREBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(DSDocREBert, self).__init__()

        self.bert_hidden = 768
        self.rel_num = config.getint('train', 'rel_num')

        self.hidden = 256
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Bilinear(self.bert_hidden, self.bert_hidden, self.hidden)
        self.binaryLinear = nn.Linear(self.hidden, 1)
        self.rank_score = nn.Linear(self.hidden, 1)

        self.tasks = set(config.get('train', 'tasks').split(','))

        self.men_bilinar = nn.Bilinear(self.bert_hidden, self.bert_hidden, 1)

        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def init_multi_gpu(self, device, config, *args, **params):
        return
    
    def save_pretrained(self, path):
        self.bert.save_pretrained(path)


    def forward_RelDet(self, ent, head, tail, rdlabel, acc_result, constant):
        # head: batch, rank_num, neg_num
        # ent: batch, ent_num, bert_hidden
        batch = constant['batch']
        rank_num = head.shape[1]
        neg_num = head.shape[2]

        indice = torch.arange(0, batch).view(batch, 1, 1).repeat(1, rank_num, neg_num)
        rdhead = ent[indice, head].view(batch, rank_num, neg_num, self.bert_hidden)
        rdtail = ent[indice, tail].view(batch, rank_num, neg_num, self.bert_hidden)

        feature = self.linear(rdhead, rdtail) # batch, rank_num, neg_num, rel_num

        score = self.rank_score(feature).squeeze(3).view(batch * rank_num, neg_num) # batch, rank_num, neg_num
        rdlabel = rdlabel.view(batch * rank_num)
        rdloss = self.loss(score, rdlabel)
        acc_result['RelDet'] = softmax_accu(score, rdlabel, acc_result['RelDet'])
        
        return rdloss, acc_result
    
    def forward_RelAlign(self, queryent, candent, queryhead, querytail, candhead, candtail, ralabel, acc_result, constant):
        batch = constant['batch']
        sample_num = constant['sample_num']

        qindice = torch.arange(0, batch)
        qhead = queryent[qindice, queryhead].view(batch, self.bert_hidden)
        qtail = queryent[qindice, querytail].view(batch, self.bert_hidden)
        qfeature = self.linear(qhead, qtail) # batch, rel_num


        cindice = torch.arange(0, batch).view(batch, 1).repeat(1, sample_num)
        chead = candent[cindice, candhead].view(batch, sample_num, self.bert_hidden)
        ctail = candent[cindice, candtail].view(batch, sample_num, self.bert_hidden)
        cfeature = self.linear(chead, ctail) # batch, sample_num, rel_num

        score = self.binaryLinear(torch.pow(qfeature.unsqueeze(1) - cfeature, 2)).squeeze(2) # batch, sample_num
        raloss = self.loss(score, ralabel)
        acc_result['RelAlign'] = softmax_accu(score, ralabel, acc_result['RelAlign'])

        return raloss, acc_result

    def forward_MenMat(self, ent, hiddens, query, cands, mmLabel, acc_result, constant):
        batch = constant['batch']
        men_sample = query.shape[1]
        neg_num = cands.shape[2]

        qindice = torch.arange(0, batch).view(batch, 1).repeat(1, men_sample)
        qmen = hiddens[qindice, query].view(batch, men_sample, self.bert_hidden)

        cindice = torch.arange(0, batch).view(batch, 1, 1).repeat(1, men_sample, neg_num)
        candmen = hiddens[cindice, cands].view(batch, men_sample, neg_num, self.bert_hidden)

        score = self.men_bilinar(qmen.unsqueeze(2).repeat(1, 1, neg_num, 1), candmen).squeeze(3).view(batch * men_sample, neg_num)
        mmlabel = mmLabel.view(batch * men_sample)
        mmloss = self.loss(score, mmlabel)
        acc_result['MenMat'] = softmax_accu(score, mmlabel, acc_result['MenMat'])
        
        return mmloss, acc_result
    
    def forward_XMenMat(self, ent1, ent2, query, cand, label, acc_result, constant):
        batch = constant['batch']
        neg_num = cand.shape[1]

        qindice = torch.arange(0, batch)
        qent = ent1[qindice, query].view(batch, self.bert_hidden)

        cindice = torch.arange(0, batch).view(batch, 1).repeat(1, neg_num)
        candent = ent2[cindice, cand].view(batch, neg_num, self.bert_hidden)

        score = self.men_bilinar(qent.unsqueeze(1).repeat(1, neg_num, 1), candent).squeeze(2).view(batch, neg_num)
        xmmloss = self.loss(score, label)
        acc_result['XMenMat'] = softmax_accu(score, label, acc_result['XMenMat'])

        return xmmloss, acc_result

    def forward_XRelDet(self, ent1, ent2, crh1, crh2, crt1, crt2, xrdlabel, acc_result, constant):
        # crh1: batch, label_num, neg_num
        batch = constant['batch']
        label_num = crh1.shape[1]
        neg_num = crh1.shape[2]

        indice = torch.arange(0, batch).view(batch, 1, 1).repeat(1, label_num, neg_num)
        crhead1 = ent1[indice, crh1].view(batch, label_num, neg_num, self.bert_hidden)
        crtail1 = ent1[indice, crt1].view(batch, label_num, neg_num, self.bert_hidden)
        crhead2 = ent2[indice, crh2].view(batch, label_num, neg_num, self.bert_hidden)
        crtail2 = ent2[indice, crt2].view(batch, label_num, neg_num, self.bert_hidden)

        fea1 = self.linear(crhead1, crtail1) # batch, label_num, neg_num, hidden
        fea2 = self.linear(crhead2, crtail2)

        score1 = self.rank_score(fea1).squeeze(3)
        score2 = self.rank_score(fea2).squeeze(3)

        score = torch.cat([score1, score2], dim = 2).view(batch * label_num, neg_num * 2)
        xrdlabel = xrdlabel.view(batch * label_num)
        xrdloss = self.loss(score, xrdlabel)
        acc_result['XRelDet'] = softmax_accu(score, xrdlabel, acc_result['XRelDet'])

        return xrdloss, acc_result


    def forward(self, data, config, gpu_list, acc_result, mode):
        if acc_result is None:
            acc_result = {}
            for key in self.tasks:
                acc_result[key] = {'right': 0, 'all': 0}
        constant = {}
        batch = constant['batch'] = data['doc1'].shape[0]
        constant['sample_num'] = data['RelAlign_CandHead1'].shape[1]
        

        hidden1, _ = self.bert(data['doc1'], encoder_attention_mask = data['attMask1'])
        hidden2, _ = self.bert(data['doc2'], encoder_attention_mask = data['attMask2'])
        

        # pos: batch, ent_num, men_num
        #from IPython import embed; embed()
        ent_num, men_num = data['entPos1'].shape[1], data['entPos1'].shape[2]
        indice = torch.arange(0, batch).view(batch, 1, 1).repeat(1, ent_num, men_num)
        ent1 = hidden1[indice, data['entPos1']].view(batch, ent_num, men_num, self.bert_hidden)
        ent2 = hidden2[indice, data['entPos2']].view(batch, ent_num, men_num, self.bert_hidden)

        ent1 = torch.max(ent1, dim = 2)[0]
        ent2 = torch.max(ent2, dim = 2)[0]

        loss = 0
        if 'RelAlign' in self.tasks:
            mtbloss1, acc_result = self.forward_RelAlign(ent1, ent2, data['RelAlign_QueryHead1'], data['RelAlign_QueryTail1'], data['RelAlign_CandHead2'], data['RelAlign_CandTail2'], data['RelAlign_Label12'], acc_result, constant)
            mtbloss2, acc_result = self.forward_RelAlign(ent2, ent1, data['RelAlign_QueryHead2'], data['RelAlign_QueryTail2'], data['RelAlign_CandHead1'], data['RelAlign_CandTail1'], data['RelAlign_Label21'], acc_result, constant)
            mtbloss = mtbloss1 + mtbloss2
            loss += mtbloss

        if 'RelDet' in self.tasks:
            rankloss1, acc_result = self.forward_RelDet(ent1, data['RelDet_Head1'], data['RelDet_Tail1'], data['RelDet_Label1'], acc_result, constant)
            rankloss2, acc_result = self.forward_RelDet(ent2, data['RelDet_Head2'], data['RelDet_Tail2'], data['RelDet_Label2'], acc_result, constant)
            rankloss = rankloss1 + rankloss2
            loss += rankloss

        if 'MenMat' in self.tasks:
            menloss1, acc_result = self.forward_MenMat(ent1, hidden1, data['MenMat_Query1'], data['MenMat_Cand1'], data['MenMat_Label1'], acc_result, constant)
            menloss2, acc_result = self.forward_MenMat(ent2, hidden2, data['MenMat_Query2'], data['MenMat_Cand2'], data['MenMat_Label2'], acc_result, constant)
            menloss = menloss1 + menloss2
            loss += menloss

        if 'XMenMat' in self.tasks:
            cmenloss1, acc_result = self.forward_XMenMat(ent1, ent2, data['XMenMat_Query1'], data['XMenMat_Cand2'], data['XMenMat_Label2'], acc_result, constant)
            cmenloss2, acc_result = self.forward_XMenMat(ent2, ent1, data['XMenMat_Query2'], data['XMenMat_Cand1'], data['XMenMat_Label1'], acc_result, constant)
            cmenloss = cmenloss1 + cmenloss2
            loss += cmenloss

        if 'XRelDet' in self.tasks:
            crankloss, acc_result = self.forward_XRelDet(ent1, ent2, data['XRelDet_Head1'], data['XRelDet_Head2'], data['XRelDet_Tail1'], data['XRelDet_Tail2'], data['XRelDet_Label'], acc_result, constant)
            loss += crankloss

        if torch.isnan(loss):
            from IPython import embed; embed()

        return {"loss": loss, 'acc_result': acc_result}

