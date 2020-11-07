import json
import torch
import os
import numpy as np
from collections import defaultdict
import random
from formatter.Basic import BasicFormatter
import nltk
from transformers import BertTokenizer
import random

class PreDenoiseFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.mode = mode
        self.doc_len = 512
        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.pos_sample = 3
        self.neg_sample = 5
        self.ent_num = 42

    def encode_doc(self, doc):
        ss = [[self.tokenizer.tokenize(w) for w in s] for s in doc['sents']]

        for entid, ent in enumerate(doc['vertexSet']):
            entrep = random.random()
            for men in ent:
                ss[men['sent_id']][men['pos'][0]].insert(0, '[unused%d]' % (entid * 2 + 1))
                ss[men['sent_id']][men['pos'][1] - 1].append('[unused%d]' % (entid * 2 + 2))
        
        wordP = []
        docs = ['[CLS]']
        for s in ss:
            wordP.append([])
            for w in s:
                wordP[-1].append(len(docs))
                docs += w
        if len(docs) < 512:
            docs.append('[SEP]')
            bert_mask = [1] * len(docs) + [0] * (512 - len(docs))
            docs += (512 - len(docs)) * ['[PAD]']
        else:
            docs = docs[:511] + ['[SEP]']
            bert_mask = [1] * 512
        menNum = 3
        poses = [[wordP[men['sent_id']][men['pos'][0]] for men in ent if wordP[men['sent_id']][men['pos'][0]] < 512][:menNum] for ent in doc['vertexSet']]
        for i in range(len(poses)):
            if len(poses[i]) == 0:
                poses[i] = [0] * menNum
            poses[i] = poses[i] + [poses[i][0]] * (menNum - len(poses[i]))
        while len(poses) < self.ent_num:
            poses.append([0] * menNum)

        return self.tokenizer.convert_tokens_to_ids(docs), bert_mask, poses
    
    def encode_rank(self, doc, poses):
        positive = [(la['h'], la['t']) for la in doc['labels']]
        negative = [(i, j) for i in range(len(doc['vertexSet'])) for j in range(len(doc['vertexSet'])) if i != j and (i, j) not in positive]
        while len(negative) < self.neg_sample:
            negative += negative
        
        rankh = []
        rankt = []
        ranklabel = []
        for i in range(self.pos_sample):
            rankh.append([])
            rankt.append([])

            ppair = random.choice(positive)
            npair = random.sample(negative, self.neg_sample)
            pl = random.randint(0, self.neg_sample)
            ranklabel.append(pl)
            for pair in (npair[:pl] + [ppair] + npair[pl:]):
                rankh[-1].append(pair[0])
                rankt[-1].append(pair[1])
        return rankh, rankt, ranklabel
    
    def process_test(self, data, config):
        docs = []
        attMask = []
        entPosition = []
        heads = []
        tails = []
        for da in data:
            doc, mask, pos = self.encode_doc(da)
            docs.append(doc)
            attMask.append(mask)
            entPosition.append(pos)

            pairs = [(head, tail) for head in range(len(da['vertexSet'])) for tail in range(len(da['vertexSet'])) if head != tail]
            pairs += [(0, 0)] * (self.ent_num * (self.ent_num - 1) - len(pairs))
            heads.append([p[0] for p in pairs])
            tails.append([p[1] for p in pairs])
        return {
            'doc': torch.LongTensor(docs),
            'attMask': torch.FloatTensor(attMask),
            'entPos': torch.LongTensor(entPosition),
            'heads': torch.LongTensor(heads),
            'tails': torch.LongTensor(tails),
            'titles': [da['title'] for da in data],
        }


    def process(self, data, config, mode, *args, **params):
        if mode == 'test':
            return self.process_test(data, config)
        docs = []
        attMask = []
        entPosition = []

        rankh = []
        rankt = []
        ranklabel = []

        for da in data:
            doc, mask, pos = self.encode_doc(da)
            rh, rt, rl = self.encode_rank(da, pos)

            docs.append(doc)
            attMask.append(mask)
            entPosition.append(pos)

            rankh.append(rh)
            rankt.append(rt)
            ranklabel.append(rl)
        
        return {
            'doc': torch.LongTensor(docs),
            'attMask': torch.FloatTensor(attMask),
            'entPos': torch.LongTensor(entPosition),
            'rankh': torch.LongTensor(rankh),
            'rankt': torch.LongTensor(rankt),
            'ranklabel': torch.LongTensor(ranklabel),
        }
            

