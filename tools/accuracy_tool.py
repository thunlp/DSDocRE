import logging
import torch

logger = logging.Logger(__name__)



def Multi_Label_DocRED(outputs, label, label_mask, result = None):
    if result is None:
        result = {'right': 0, 'pre_num': 0, 'actual_num': 0, 'auc_item': []}

    if len(label[0]) != len(outputs[0]):
        raise ValueError('Input dimensions of labels and outputs must match.')

    score, id1 = torch.max(outputs, dim = 1) # batch, label_num
    
    # id2 = torch.max(label, dim = 1)[1]
    for a in range(len(id1)):
        it_is = int(id1[a])

        if int(label_mask[a]) == 0:
            continue

        if int(label[a][96]) == 0:
            result['actual_num'] += int(label[a].sum())
        
        if it_is == 96:
            continue

        result['pre_num'] += 1
        result['auc_item'].append( (int(int(label[a][it_is]) == 1), float(score[a])) )

        if int(label[a][it_is]) == 1:
            result['right'] += 1
    return result
    '''
    for docid in range(out.shape[0]):
        for lid in range(out.shape[1]):
            it_is = int(out[docid, lid])
            if int(label_mask[docid, lid]) == 0:
                continue
            if int(label[docid, lid, 96]) == 0:
                result['actual_num'] += 1
            if not it_is == 96:
                result['pre_num'] += 1
            else:
                continue
            
            if int(label[docid, lid, it_is]) == 1:
                result['right'] += 1
    return result
    '''


def binary_accu(out, labels, acc_result):
    predict = (out > 0).int()
    right = int(torch.sum((predict == labels).int()))
    acc_result['right'] += right
    acc_result['all'] += int(out.shape[0])
    return acc_result

def rank_accu(scores, posInd, acc_result):
    _, predict = torch.topk(scores, dim = 1, k = posInd.shape[1])
    pre = predict.tolist()
    pos = posInd.tolist()
    for i in range(posInd.shape[0]):
        acc_result['right'] = len(set(pre[i]) & set(pos[i]))
    acc_result['all'] += int(posInd.shape[0] * posInd.shape[1])
    return acc_result

def softmax_accu(out, labels, acc_result):
    predict = torch.max(out, dim = 1)[1]
    right = int(torch.sum((predict == labels).int()))
    acc_result['right'] += right
    acc_result['all'] += int(out.shape[0]) - int(torch.sum(labels == -100).int())
    return acc_result

