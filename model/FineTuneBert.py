import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.accuracy_tool import *
from transformers import BertModel

class FineTuneBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(FineTuneBert, self).__init__()

        self.hidden = config.getint('model', 'hidden_size')
        self.rel_num = config.getint('train', 'rel_num')
        self.encoder = BertModel.from_pretrained(config.get("model", "bert_path"))
        
        #self.linear = nn.Linear(768, self.hidden)
        #self.hidden = 768
        self.bert_hidden = 768
        hidden = 256
        self.bilinear = nn.Bilinear(self.bert_hidden, self.bert_hidden, hidden)
        self.classify = nn.Linear(hidden, self.rel_num)
        
        self.train_loss = nn.BCEWithLogitsLoss(reduction='none')

    
    def forward(self, data, config, gpu_list, acc_result, mode):
        docs = data['docs'] # batch, max_len

        head = data['head'] # batch, label_num, 3
        tail = data['tail']

        batch = docs.shape[0]
        label_num = head.shape[1]
        max_len = docs.shape[1]
        men_num = head.shape[2]

        docs, _ = self.encoder(docs, data['bert_mask'].float()) # batch, max_len, bert_hidden


        headrep = docs[[[[i] * men_num] * label_num for i in range(batch)], head].view(batch, label_num, men_num, self.bert_hidden)
        tailrep = docs[[[[i] * men_num] * label_num for i in range(batch)], tail].view(batch, label_num, men_num, self.bert_hidden)

        headrep = torch.max(headrep, dim = 2)[0]
        tailrep = torch.max(tailrep, dim = 2)[0]
        out = self.bilinear(headrep, tailrep)

        out = self.classify(out)

        if mode != 'test':
            labels = data['labels'] # batch, label_num, rel_num
            label_mask = data['label_mask']
            '''
            loss2 = self.train_loss(gate.squeeze(2), 1 - labels[:,:,-1].float())
            loss2 = torch.sum(loss2 * label_mask) / torch.sum(label_mask)
            '''
            labels = labels.view(batch * labels.shape[1], labels.shape[2])

            y = out.view(batch * out.shape[1], out.shape[2])

            label_mask = label_mask.view(batch * out.shape[1]) # batch, label_num

            loss = self.train_loss(y, labels.float()) # batch, label_num, rel_num
            loss = torch.sum(loss * label_mask.unsqueeze(1)) / (torch.sum(label_mask) * self.rel_num)


            acc_result = Multi_Label_DocRED(y, labels, label_mask, acc_result)
            if mode == 'valid':
                acc_result['actual_num'] = 12323
        else:
            res = torch.max(out, dim = 2)[1]
            return {'res': res.cpu().tolist(), 'pair': data['pairs'], 'title': data['titles']}
        return {'loss': loss, 'acc_result': acc_result}

