import json
import torch
import os
import numpy as np
import random
from formatter.Basic import BasicFormatter
import nltk
from transformers import BertTokenizer
import random


class PreTrainFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))

        self.doc_len = 512
        self.blank_prob = 0.8
        self.sample_num = 10
        
        self.neg_sample = 8
        self.pos_sample = 5
        
        self.men_sample_num = 3
        self.ent_num = 42

        self.intra_rel_pos_sample = 3

    def encode_candidate(self, doc, dht):
        if dht == (0, 0):
            return [0] * (self.pos_sample + self.neg_sample + 1), [0] * (self.pos_sample + self.neg_sample + 1), -100
        pos = {(la['h'], la['t']): la['r'] for la in doc['labels'] if la['good']}
        disPosPair = set([(la['h'], la['t']) for la in doc['labels']])
        negative = [(i, j) for i in range(len(doc['vertexSet'])) for j in range(len(doc['vertexSet'])) if i != j and (i, j) not in disPosPair]
        while len(negative) < self.neg_sample:
            negative += negative

        posPairs = [pair for pair in pos if pos[pair] != pos[dht]]
        if len(posPairs) == 0:
            posSample = random.sample(negative, self.pos_sample)
        else:
            while len(posPairs) < self.pos_sample:
                posPairs += posPairs
            posSample = random.sample(posPairs, self.pos_sample)

        negSample = random.sample(negative, self.neg_sample)

        candh = []
        candt = []

        sample_data = [(pair, 'p') for pair in posSample] + [(pair, 'n') for pair in negSample]
        random.shuffle(sample_data)
        label = random.randint(0, len(sample_data))
        for pairid, pair in enumerate(sample_data[:label] + [(dht, 'g')] + sample_data[label:]):
            candh.append(pair[0][0])
            candt.append(pair[0][1])
        return candh, candt, label

    def encode_rank(self, doc):
        pos = {(la['h'], la['t']): la['r'] for la in doc['labels'] if la['good']}
        disPosPair = set([(la['h'], la['t']) for la in doc['labels']])
        negative = [(i, j) for i in range(len(doc['vertexSet'])) for j in range(len(doc['vertexSet'])) if i != j and (i, j) not in disPosPair]
        
        pos = list(pos)
        while len(negative) < self.neg_sample:
            negative += negative

        rankh = []
        rankt = []
        ranklabel = []
        if len(pos) == 0:
            rankt = rankh = [[0] * self.neg_sample] * 3
            ranklabel = [-100] * 3
            return rankh, rankt, ranklabel
        
        for i in range(3):
            rankh.append([])
            rankt.append([])

            ppair = random.choice(pos)
            pairs = random.sample(negative, self.neg_sample)
            pl = random.randint(0, self.neg_sample - 1)
            pairs[pl] = ppair
            ranklabel.append(pl)
            for pair in pairs:
                rankh[-1].append(pair[0])
                rankt[-1].append(pair[1])
        return rankh, rankt, ranklabel

    def encode_doc(self, doc, maskMen = []):
        ss = [[self.tokenizer.tokenize(w) for w in s] for s in doc['sents']]
        
        for mmen in maskMen:
            men, entid = mmen[0], mmen[1]
            for posid in range(men['pos'][0], men['pos'][1]):
                ss[men['sent_id']][posid] = []
            ss[men['sent_id']][men['pos'][0]].insert(0, '[unused200]')
        
        for entid, ent in enumerate(doc['vertexSet']):
            entrep = random.random()
            if entrep > self.blank_prob:
                for men in ent:
                    for posid in range(men['pos'][0], men['pos'][1]):
                        ss[men['sent_id']][posid] = []
                    ss[men['sent_id']][men['pos'][0]] = ['[unused0]']
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

        menpos = []
        menCand = []
        menLabel = []
        cand_num = 5
        if len(maskMen) == 0:
            menpos = [0] * self.men_sample_num
            menLabel = [-100] * self.men_sample_num
            #menCand = [[poses[0]] * cand_num] * self.men_sample_num
            menCand = [[0] * cand_num] * self.men_sample_num
        for mmen in maskMen:
            men, entid = mmen[0], mmen[1]
            cands = random.sample(range(len(doc['vertexSet'])), cand_num)
            if entid in cands:
                menLabel.append(cands.index(entid))
            else:
                menl = random.randint(0, cand_num - 1)
                cands[menl] = entid
                menLabel.append(menl)
            #menCand.append([poses[c] for c in cands])
            menCand.append(cands)

            if wordP[men['sent_id']][men['pos'][0]] < self.doc_len:
                menpos.append(wordP[men['sent_id']][men['pos'][0]])
            else:
                menpos.append(0)
                menLabel[-1] = -100

        return self.tokenizer.convert_tokens_to_ids(docs), bert_mask, poses, menpos, menCand, menLabel

    def encode_mention(self, doc):
        num = self.men_sample_num
        mm = {entid for entid, ent in enumerate(doc['vertexSet']) if len(ent) > 1}
        ents = random.sample(mm, min(num, len(mm)))
        maskMen = []
        for entid in ents:
            menid = random.randint(0, len(doc['vertexSet'][entid]) - 1)
            maskMen.append((doc['vertexSet'][entid][menid], entid))
            doc['vertexSet'][entid].pop(menid)
        if len(maskMen) < num and len(maskMen) != 0:
            maskMen += [maskMen[0]] * (num - len(maskMen))
        
        return maskMen

    def encode_intra_rel(self, doc):
        rel2ins = {}
        for la in doc['labels']:
            if not la['good']:
                continue
            if not la['r'] in rel2ins:
                rel2ins[la['r']] = []
            rel2ins[la['r']].append(la)
        for rel in [key for key in rel2ins if len(rel2ins[key]) < 2]:
            rel2ins.pop(rel)
        
        if len(rel2ins) == 0:
            label = -100
            candh = candt = [0] * (self.pos_sample + self.neg_sample)
            query = (0, 0)
        else:
            rel = random.choice(list(rel2ins.keys()))
            match_pair = random.sample(rel2ins[rel], 2)
            query = (match_pair[0]['h'], match_pair[0]['t'])

            positive = {(la['h'], la['t']): la['r'] for la in doc['labels'] if la['good']}
            disPosPair = set([(la['h'], la['t']) for la in doc['labels']])
            negative = [(i, j) for i in range(len(doc['vertexSet'])) for j in range(len(doc['vertexSet'])) if i != j and (i, j) not in disPosPair]

            posPairs = [pair for pair in positive if positive[pair] != rel]
            if len(posPairs) == 0:
                posSample = random.sample(negative, self.pos_sample)
            else:
                while len(posPairs) < self.pos_sample:
                    posPairs += posPairs
                posSample = random.sample(posPairs, self.pos_sample)
            negSample = random.sample(negative, self.neg_sample)


            sample_data = posSample + negSample
            random.shuffle(sample_data)
            assert len(sample_data) == self.neg_sample + self.pos_sample
            label = random.randint(0, len(sample_data) - 1)
            sample_data[label] = (match_pair[1]['h'], match_pair[1]['t'])

            candh = [pair[0] for pair in sample_data]
            candt = [pair[1] for pair in sample_data]
        
        return query[0], query[1], candh, candt, label

    def encode_cross_rank(self, doc1, doc2):
        pos1 = [(la['h'], la['t']) for la in doc1['labels'] if la['good']]
        disPosPair1 = set([(la['h'], la['t']) for la in doc1['labels']])
        negative1 = [(i, j) for i in range(len(doc1['vertexSet'])) for j in range(len(doc1['vertexSet'])) if i != j and (i, j) not in disPosPair1]
        while len(negative1) < self.neg_sample:
            negative1 += negative1

        pos2 = [(la['h'], la['t']) for la in doc2['labels'] if la['good']]
        disPosPair2 = set([(la['h'], la['t']) for la in doc2['labels']])
        negative2 = [(i, j) for i in range(len(doc2['vertexSet'])) for j in range(len(doc2['vertexSet'])) if i != j and (i, j) not in disPosPair2]
        while len(negative2) < self.neg_sample:
            negative2 += negative2
        

        cross_rankh1 = []
        cross_rankt1 = []
        cross_rankh2 = []
        cross_rankt2 = []
        cross_ranklabel = []
        if len(pos1) == 0 or len(pos2) == 0:
            cross_rankh1 = cross_rankh2 = cross_rankt1 = cross_rankt2 = [[0] * (self.neg_sample // 2)] * 3
            cross_ranklabel = [-100] * 3
            return cross_rankh1, cross_rankh2, cross_rankt1, cross_rankt2, cross_ranklabel

        
        for i in range(3):
            cross_rankh1.append([])
            cross_rankt1.append([])
            cross_rankh2.append([])
            cross_rankt2.append([])
            pairs1 = random.sample(negative1, self.neg_sample // 2)
            pairs2 = random.sample(negative2, self.neg_sample // 2)
            pl = random.randint(0, (self.neg_sample // 2) - 1)
            if random.random() < 0.5:
                cross_ranklabel.append(pl)
                pairs1[pl] = random.choice(pos1)
            else:
                cross_ranklabel.append(pl + self.neg_sample // 2)
                pairs2[pl] = random.choice(pos2)
            for pair in pairs1:
                cross_rankh1[-1].append(pair[0])
                cross_rankt1[-1].append(pair[1])
            for pair in pairs2:
                cross_rankh2[-1].append(pair[0])
                cross_rankt2[-1].append(pair[1])

        return cross_rankh1, cross_rankh2, cross_rankt1, cross_rankt2, cross_ranklabel

    def encode_cross_mention(self, doc1, doc2):
        name2ent = {}
        for entid, ent in enumerate(doc1['vertexSet']):
            for men in ent:
                name2ent[men['name']] = entid
        sameEnt = set()
        for entid, ent in enumerate(doc2['vertexSet']):
            for men in ent:
                if men['name'] in name2ent:
                    sameEnt.add((name2ent[men['name']], entid))
        if len(sameEnt) == 0:
            label1 = label2 = -100
            ent1 = ent2 = 0
            cand1 = cand2 = [0] * self.neg_sample
        else:
            ent1, ent2 = random.choice(list(sameEnt))
            vertex1 = list(range(len(doc1['vertexSet'])))
            vertex1.remove(ent1)
            vertex2 = list(range(len(doc2['vertexSet'])))
            vertex2.remove(ent2)
            while len(vertex1) < self.neg_sample:
                vertex1 += vertex1
            while len(vertex2) < self.neg_sample:
                vertex2 += vertex2
            cand1 = random.sample(vertex1, self.neg_sample)
            cand2 = random.sample(vertex2, self.neg_sample)
            label1 = random.randint(0, self.neg_sample - 1)
            label2 = random.randint(0, self.neg_sample - 1)
            cand1[label1] = ent1
            cand2[label2] = ent2

        return ent1, ent2, cand1, cand2, label1, label2


    def process(self, data, config, mode, *args, **params):
        inp1 = []
        inp2 = []

        entPos1 = []
        entPos2 = []
        
        attMask1 = []
        attMask2 = []
        
        # cross same rel
        RelAlign_QueryHead1 = []
        RelAlign_QueryTail1 = []
        RelAlign_CandHead1 = []
        RelAlign_CandTail1 = []
        
        RelAlign_QueryHead2 = []
        RelAlign_QueryTail2 = []
        RelAlign_CandHead2 = []
        RelAlign_CandTail2 = []

        RelAlign_Label12 = []
        RelAlign_Label21 = []

        # intra same rel
        intraQueryHead1 = []
        intraQueryTail1 = []
        intraCandHead1 = []
        intraCandTail1 = []
        intraRelLabel1 = []

        intraQueryHead2 = []
        intraQueryTail2 = []
        intraCandHead2 = []
        intraCandTail2 = []
        intraRelLabel2 = []


        # intra rank
        RelDet_Head1 = []
        RelDet_Tail1 = []
        RelDet_Label1 = []

        RelDet_Head2 = []
        RelDet_Tail2 = []
        RelDet_Label2 = []
        
        # intra mention
        MenMat_Query1 = []
        MenMat_Cand1 = []
        MenMat_Label1 = []

        MenMat_Query2 = []
        MenMat_Cand2 = []
        MenMat_Label2 = []

        # cross mention
        XMenMat_Query2 = []
        XMenMat_Cand1 = []
        XMenMat_Label1 = []

        XMenMat_Query1 = []
        XMenMat_Cand2 = []
        XMenMat_Label2 = []

        # cross rank
        XRelDet_Head1 = []
        XRelDet_Tail1 = []
        XRelDet_Head2 = []
        XRelDet_Tail2 = []
        XRelDet_Label = []

        for da in data:
            maskMen1 = self.encode_mention(da['doc1'])
            maskMen2 = self.encode_mention(da['doc2'])

            in1, ma1, pos1, mq1, mc1, ml1 = self.encode_doc(da['doc1'], maskMen1)
            in2, ma2, pos2, mq2, mc2, ml2 = self.encode_doc(da['doc2'], maskMen2)

            inp1.append(in1)
            inp2.append(in2)
            attMask1.append(ma1)
            attMask2.append(ma2)
            entPos1.append(pos1)
            entPos2.append(pos2)

            
            # ================================
            MenMat_Query1.append(mq1)
            MenMat_Cand1.append(mc1)
            MenMat_Label1.append(ml1)

            MenMat_Query2.append(mq2)
            MenMat_Cand2.append(mc2)
            MenMat_Label2.append(ml2)
            
            cmq1, cmq2, cmc1, cmc2, cml1, cml2 = self.encode_cross_mention(da['doc1'], da['doc2'])
            XMenMat_Query1.append(cmq1)
            XMenMat_Query2.append(cmq2)
            XMenMat_Cand1.append(cmc1)
            XMenMat_Cand2.append(cmc2)
            XMenMat_Label1.append(cml1)
            XMenMat_Label2.append(cml2)

            # ================================
            RelAlign_QueryHead1.append(da['ht1'][0])
            RelAlign_QueryTail1.append(da['ht1'][1])
            RelAlign_QueryHead2.append(da['ht2'][0])
            RelAlign_QueryTail2.append(da['ht2'][1])

            h2, t2, l12 = self.encode_candidate(da['doc2'], da['ht2'])
            RelAlign_CandHead2.append(h2)
            RelAlign_CandTail2.append(t2)
            RelAlign_Label12.append(l12)

            h1, t1, l21 = self.encode_candidate(da['doc1'], da['ht1'])
            RelAlign_CandHead1.append(h1)
            RelAlign_CandTail1.append(t1)
            RelAlign_Label21.append(l21)

            iqh1, iqt1, ich1, ict1, il1 = self.encode_intra_rel(da['doc1'])
            intraQueryHead1.append(iqh1)
            intraQueryTail1.append(iqt1)
            intraCandHead1.append(ich1)
            intraCandTail1.append(ict1)
            intraRelLabel1.append(il1)

            iqh2, iqt2, ich2, ict2, il2 = self.encode_intra_rel(da['doc2'])
            intraQueryHead2.append(iqh2)
            intraQueryTail2.append(iqt2)
            intraCandHead2.append(ich2)
            intraCandTail2.append(ict2)
            intraRelLabel2.append(il2)

            # ================================
            rankh1, rankt1, rankl1 = self.encode_rank(da['doc1'])
            RelDet_Head1.append(rankh1)
            RelDet_Tail1.append(rankt1)
            RelDet_Label1.append(rankl1)

            rankh2, rankt2, rankl2 = self.encode_rank(da['doc2'])
            RelDet_Head2.append(rankh2)
            RelDet_Tail2.append(rankt2)
            RelDet_Label2.append(rankl2)

            crh1, crh2, crt1, crt2, crl  = self.encode_cross_rank(da['doc1'], da['doc2'])
            XRelDet_Head1.append(crh1)
            XRelDet_Head2.append(crh2)
            XRelDet_Tail1.append(crt1)
            XRelDet_Tail2.append(crt2)
            XRelDet_Label.append(crl)
        
        ret = {
            'doc1': torch.LongTensor(inp1),
            'doc2': torch.LongTensor(inp2),
            'attMask1': torch.FloatTensor(attMask1),
            'attMask2': torch.FloatTensor(attMask2),
            'entPos1': torch.LongTensor(entPos1),
            'entPos2': torch.LongTensor(entPos2),

            'RelAlign_QueryHead1': torch.LongTensor(RelAlign_QueryHead1),
            'RelAlign_QueryTail1': torch.LongTensor(RelAlign_QueryTail1),
            'RelAlign_CandHead1': torch.LongTensor(RelAlign_CandHead1),
            'RelAlign_CandTail1': torch.LongTensor(RelAlign_CandTail1),
            
            'RelAlign_QueryHead2': torch.LongTensor(RelAlign_QueryHead2),
            'RelAlign_QueryTail2': torch.LongTensor(RelAlign_QueryTail2),
            'RelAlign_CandHead2': torch.LongTensor(RelAlign_CandHead2),
            'RelAlign_CandTail2': torch.LongTensor(RelAlign_CandTail2),

            'RelAlign_Label12': torch.LongTensor(RelAlign_Label12),
            'RelAlign_Label21': torch.LongTensor(RelAlign_Label21),

            'RelDet_Head1': torch.LongTensor(RelDet_Head1),
            'RelDet_Tail1': torch.LongTensor(RelDet_Tail1),
            'RelDet_Label1': torch.LongTensor(RelDet_Label1),

            'RelDet_Head2': torch.LongTensor(RelDet_Head2),
            'RelDet_Tail2': torch.LongTensor(RelDet_Tail2),
            'RelDet_Label2': torch.LongTensor(RelDet_Label2),

            'MenMat_Query1': torch.LongTensor(MenMat_Query1),
            'MenMat_Cand1': torch.LongTensor(MenMat_Cand1),
            'MenMat_Label1': torch.LongTensor(MenMat_Label1),

            'MenMat_Query2': torch.LongTensor(MenMat_Query2),
            'MenMat_Cand2': torch.LongTensor(MenMat_Cand2),
            'MenMat_Label2': torch.LongTensor(MenMat_Label2),

            'XMenMat_Query1': torch.LongTensor(XMenMat_Query1),
            'XMenMat_Cand1': torch.LongTensor(XMenMat_Cand1),
            'XMenMat_Label1': torch.LongTensor(XMenMat_Label1),

            'XMenMat_Query2': torch.LongTensor(XMenMat_Query2),
            'XMenMat_Cand2': torch.LongTensor(XMenMat_Cand2),
            'XMenMat_Label2': torch.LongTensor(XMenMat_Label2),

            'XRelDet_Head1': torch.LongTensor(XRelDet_Head1),
            'XRelDet_Head2': torch.LongTensor(XRelDet_Head2),
            'XRelDet_Tail1': torch.LongTensor(XRelDet_Tail1),
            'XRelDet_Tail2': torch.LongTensor(XRelDet_Tail2),
            'XRelDet_Label': torch.LongTensor(XRelDet_Label),

        }

        return ret


