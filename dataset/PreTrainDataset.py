import json
import os
from torch.utils.data import Dataset
import random
import numpy as np

class PreTrainDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        if config.getboolean('train', 'no_valid') and mode != 'train':
            return
        self.load_score(config)
        self.load_doc_data(config)
        
        #self.rels = list(self.rel2doc.keys())
        self.pairs = list(self.pair2doc.keys())

        self.same_ratio = 0.6

        try:
            train_steps = config.getint('train', 'train_steps')
        except:
            train_steps = 1

        if mode != 'train':
            self.train_num = 200 * config.getint('train', 'batch_size') * train_steps
        else:
            self.train_num = 1000 * config.getint('train', 'batch_size') * train_steps

    def load_score(self, config):
        score_path = config.get("data", "doc_score_path")
        self.score = np.load(os.path.join(score_path, 'train_distant_score.npy'))
        title = json.load(open(os.path.join(score_path, 'train_distant_title.json'), 'r'))
        self.title2id = {t: i for i, t in enumerate(title)}

    def load_doc_data(self, config):
        doc_data_path = config.get("data", "doc_data_path")
        
        self.doc_data = json.load(open(doc_data_path, 'r'))
        good = set()
        for docid, doc in enumerate(self.doc_data):
            if len(doc['vertexSet']) < 5:
                continue
            score = self.score[self.title2id[doc['title']]]
            pair2score = []
            for i in range(len(doc['vertexSet'])):
                for j in range(len(doc['vertexSet'])):
                    if i == j:
                        continue
                    pair2score.append(((i, j), score[len(pair2score)]))
            pair2score.sort(key = lambda x: x[1], reverse = True)
            highscore = set([item[0] for item in pair2score[:30]])
            
            for laid, la in enumerate(doc['labels']):
                if (la['h'], la['t']) in highscore:
                    good.add((docid, la['h'], la['t']))
                    self.doc_data[docid]['labels'][laid]['good'] = True
                else:
                    self.doc_data[docid]['labels'][laid]['good'] = False

        pair2doc = json.load(open('/data/disk2/private/xcj/DocRed/data/pair2distantdoc.json', 'r'))
        
        try:
            ratio = config.getfloat('train', 'data_ratio')
        except:
            ratio = 1

        self.rest_doc = set(random.sample(range(len(self.doc_data)), int(ratio * len(self.doc_data))))

        self.pair2doc = {}
        num = 0
        for pair in pair2doc:
            instance = []
            for ins in pair2doc[pair]:
                if (tuple(ins) in good) and (ins[0] in self.rest_doc):
                    instance.append(ins)
            if len(instance) > 1:
                self.pair2doc[pair] = instance
                num += len(instance)
        
        self.rest_doc = list(self.rest_doc)
        print(len(self.pair2doc), num)

    
    def __getitem__(self, item):
        if random.random() < self.same_ratio:
            pair = random.choice(self.pairs)
            query, doc = random.sample(self.pair2doc[pair], 2)

            ret = {
                'doc1': self.doc_data[query[0]],
                'doc2': self.doc_data[doc[0]],
                'ht1': (query[1], query[2]),
                'ht2': (doc[1], doc[2]),
            }
        else:
            query, doc = random.sample(self.rest_doc, 2)
            query, doc = self.doc_data[query], self.doc_data[doc]
            while len(query['vertexSet']) < 5 or len(doc['vertexSet']) < 5 or len(query['labels']) == 0 or len(doc['labels']) == 0:
                query, doc = random.sample(self.doc_data, 2)
            ret = {
                'doc1': query,
                'doc2': doc,
                'ht1': (0, 0),
                'ht2': (0, 0),
            }
        return ret


    def __len__(self):
        return self.train_num

