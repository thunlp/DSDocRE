import json
import torch
import os
import numpy as np
from collections import defaultdict
import random
from formatter.Basic import BasicFormatter
import nltk
from transformers import BertTokenizer


class FineTuneFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.mode = mode

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.doc_len = 512
        self.label2id = json.load(open(config.get('data', 'label2id'), 'r'))
        self.label2id['NA'] = len(self.label2id)

        self.rel_num = config.getint('train', 'rel_num')
        self.ent_num = -1

        self.load_score(config)
    
    def load_score(self, config):
        path = config.get('data', '%s_score_path' % self.mode)
        scorepath = os.path.join(path, 'score.npy')
        titlepath = os.path.join(path, 'title.json')
        
        titles = json.load(open(titlepath, 'r'))
        self.scores = np.load(scorepath)
        self.title2id = {titles[i]: i for i in range(len(titles))}

    def encode_doc_ent_without_score(self, ents, labels, sents, title):
        ent_num = len(ents)
        ss = [[self.tokenizer.tokenize(w) for w in s] for s in sents]
        
        for i, ent in enumerate(ents):
            for men in ent:
                ss[men['sent_id']][men['pos'][0]].insert(0, '[unused%d]' % (2 * i + 1))
                ss[men['sent_id']][men['pos'][1] - 1].append('[unused%d]' % (2 * i + 2))
            
        wordP = []
        bert_mask = []
        docs = ['[CLS]']
        for s in ss:
            wordP.append([])
            for w in s:
                wordP[-1].append(len(docs))
                docs += w
        wordP.append([len(docs)])
        if len(docs) < 512:
            docs.append('[SEP]')
            bert_mask.append([1] * len(docs) + [0] * (512 - len(docs)))
            docs += (512 - len(docs)) * ['[PAD]']
        else:
            docs = docs[:511] + ['[SEP]']
            bert_mask.append([1] * 512)
        
        menNum = 3
        poses = [[wordP[men['sent_id']][men['pos'][0]] for men in ent if wordP[men['sent_id']][men['pos'][0]] < 512][:menNum] for ent in ents]
        for i in range(len(poses)):
            if len(poses[i]) == 0:
                poses[i] = [0] * menNum
            poses[i] = poses[i] + [poses[i][0]] * (menNum - len(poses[i]))
        
        pair2label = np.zeros((ent_num, ent_num, self.rel_num))
        pair2label[:ent_num, :ent_num, self.label2id['NA']] = 1
        good = set()
        if not labels is None:
            for l in labels:
                good.add((l['h'], l['t']))
                pair2label[l['h'], l['t'], self.label2id[l['r']]] = 1
                pair2label[l['h'], l['t'], self.label2id['NA']] = 0
        negative = [(i, j) for i in range(ent_num) for j in range(ent_num) if not(i == j and (i, j) in good)]

        head = []
        tail = []
        labels = []
        inputPair = []
        
        if self.mode == 'train':
            ttt = 90
            sample = list(good) + random.sample(negative, min(len(good) * 3, ttt - len(good), len(negative)))
        else:
            ttt = 1800
            sample = negative

        for pair in sample:
            inputPair.append(pair)
            i = pair[0]
            j = pair[1]
            head.append(poses[i])
            tail.append(poses[j])
            labels.append(pair2label[i,j])
        
        
        label_mask = [1] * len(labels) + [0] * (ttt - len(labels))
        labels += [np.zeros(self.rel_num)]  * (ttt - len(labels))
        bert_mask += [[0] * 512] * (ttt - len(labels))
        head += [[0] * menNum] * (ttt - len(head))
        tail += [[0] * menNum] * (ttt - len(tail))
        '''
        label_rel = np.zeros((ttt, ttt))
        for i in range(min(lnum, len(pairScore))):
            for j in range(i + 1, min(lnum, len(pairScore))):
                if len({pairScore[i][0][0], pairScore[i][0][1], pairScore[j][0][0], pairScore[j][0][1]}) != 4:
                    label_rel[i,j] = label_rel[j,i] = 1
        '''
        return self.tokenizer.convert_tokens_to_ids(docs), labels, head, tail, label_mask, bert_mask, inputPair#, label_rel

    def encode_doc_ent(self, ents, labels, sents, title):
        ent_num = len(ents)
        lnum = min(2*ent_num, 60)

        ss = [[self.tokenizer.tokenize(w) for w in s] for s in sents]

        scores = self.scores[self.title2id[title]]
        pairScore = []
        for i in range(ent_num):
            for j in range(ent_num):
                if i == j:
                    continue
                pairScore.append(((i, j), scores[len(pairScore)]))
        pairScore.sort(reverse=True, key=lambda x: x[1])

        entSet = {}
        for pair in pairScore[:lnum]:
            if not pair[0][0] in entSet:
                entSet[pair[0][0]] = len(entSet)
            if not pair[0][1] in entSet:
                entSet[pair[0][1]] = len(entSet)
        
        for i, ent in enumerate(entSet):
            for men in ents[ent]:
                ss[men['sent_id']][men['pos'][0]].insert(0, '[unused%d]' % (2 * i))
                ss[men['sent_id']][men['pos'][1] - 1].append('[unused%d]' % (2 * i + 1))
            
        wordP = []
        docs = ['[CLS]']
        for s in ss:
            wordP.append([])
            for w in s:
                wordP[-1].append(len(docs))
                docs += w
        wordP.append([len(docs)])
        if len(docs) < 512:
            docs.append('[SEP]')
            bert_mask = [1] * len(docs) + [0] * (512 - len(docs))
            docs += (512 - len(docs)) * ['[PAD]']
        else:
            docs = docs[:511] + ['[SEP]']
            bert_mask = [1] * 512
        
        menNum = 3
        poses = [[wordP[men['sent_id']][men['pos'][0]] for men in ent if wordP[men['sent_id']][men['pos'][0]] < 512][:menNum] for ent in ents]
        for i in range(len(poses)):
            if len(poses[i]) == 0:
                poses[i] = [0] * menNum
            poses[i] = poses[i] + [poses[i][0]] * (menNum - len(poses[i]))
        
        pair2label = np.zeros((ent_num, ent_num, self.rel_num))
        pair2label[:ent_num, :ent_num, self.label2id['NA']] = 1
        if not labels is None:
            for l in labels:
                pair2label[l['h'], l['t'], self.label2id[l['r']]] = 1
                pair2label[l['h'], l['t'], self.label2id['NA']] = 0

        head = []
        tail = []
        labels = []
        inputPair = []
        
        for pair in pairScore[:lnum]:
            inputPair.append(pair[0])
            i = pair[0][0]
            j = pair[0][1]
            head.append(poses[i])
            tail.append(poses[j])
            labels.append(pair2label[i,j])
        
        ttt = 60
        label_mask = [1] * len(labels) + [0] * (ttt - len(labels))
        labels += [np.zeros(self.rel_num)]  * (ttt - len(labels))
        head += [[0] * menNum] * (ttt - len(head))
        tail += [[0] * menNum] * (ttt - len(tail))
        '''
        label_rel = np.zeros((ttt, ttt))
        for i in range(min(lnum, len(pairScore))):
            for j in range(i + 1, min(lnum, len(pairScore))):
                if len({pairScore[i][0][0], pairScore[i][0][1], pairScore[j][0][0], pairScore[j][0][1]}) != 4:
                    label_rel[i,j] = label_rel[j,i] = 1
        '''
        return self.tokenizer.convert_tokens_to_ids(docs), labels, head, tail, label_mask, bert_mask, inputPair#, label_rel


    def process(self, data, config, mode, *args, **params):
        docs = []

        labels = []
        label_mask = []
        bert_mask = []

        head = []
        tail = []
        titles = [da['title'] for da in data]
        pairs = []

        label_rel = []
        for da in data:
            if mode == 'test':
                da['labels'] = None
            #doc_tmp, label_tmp, head_tmp, tail_tmp, mask_tmp, bert_mask_tmp, pair_tmp, rel_tmp = self.encode_doc_ent(da['vertexSet'], da['labels'], da['sents'], da['title'])
            doc_tmp, label_tmp, head_tmp, tail_tmp, mask_tmp, bert_mask_tmp, pair_tmp = self.encode_doc_ent(da['vertexSet'], da['labels'], da['sents'], da['title'])
            #doc_tmp, label_tmp, head_tmp, tail_tmp, mask_tmp, bert_mask_tmp, pair_tmp = self.encode_doc_ent_without_score(da['vertexSet'], da['labels'], da['sents'], da['title'])
            docs.append(doc_tmp)
            labels.append(label_tmp)
            label_mask.append(mask_tmp)
            head.append(head_tmp)
            tail.append(tail_tmp)
            bert_mask.append(bert_mask_tmp)
            pairs.append(pair_tmp)
            #label_rel.append(rel_tmp)
            

        return {
            'docs': torch.LongTensor(docs),
            'head': torch.LongTensor(head),
            'tail': torch.LongTensor(tail),
            #'head': torch.FloatTensor(head),
            #'tail': torch.FloatTensor(tail),

            'labels': torch.LongTensor(labels),
            'label_mask': torch.LongTensor(label_mask),
            'bert_mask': torch.LongTensor(bert_mask),
            #'label_rel': torch.LongTensor(label_rel),
            'titles': titles,
            'pairs': pairs,
        }
                        
