import logging
import torch

logger = logging.Logger(__name__)

def get_prf(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_micro_macro_result(res):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_prf(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3)
    }



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

