import json

from .accuracy_tool import gen_micro_macro_result
from sklearn import metrics
import numpy as np

def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)

def ConsGraphOutputFunc(data, config, *args, **params):
    if data['pre_num'] != 0 and data['actual_num'] != 0:
        pre = data['right'] / data['pre_num']
        recall = data['right'] / data['actual_num']
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0
    metric = {
            'precision': round(pre, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    if params['mode'] != 'train' and data['pre_num'] != 0 and data['actual_num'] != 0:
        data['auc_item'].sort(key = lambda x: x[1], reverse = True)
        pr_x = []
        pr_y = []
        correct = 0
        for i, item in enumerate(data['auc_item']):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / data['pre_num'])
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        auc = metrics.auc(x = pr_x, y = pr_y)
        metric['auc'] = round(auc, 4)
    ret = {
        'metric': metric,
        'static': {
            'right': data['right'],
            'pre_num': data['pre_num'],
            'actual_num': data['actual_num']
        }
    }
    return json.dumps(ret, sort_keys=True)

def RankOutputFunc(data, config, *args, **params):
    if not 'RP' in data:
        return basic_output_function(data, config, *args, **params)
    if data['RP'] != 0 and data['PP'] != 0:
        pre = data['TP'] / data['PP']
        recall = data['TP'] / data['RP']
    else:
        pre = recall = 0
    ret = {'pre': round(pre, 4), 'recall': round(recall, 4)}
    return json.dumps(ret)

def BinaryOutputFunc(data, config, *args, **params):
    if data['all'] == 0:
        return "0"
    else:
        return str(data['right']/data['all'])

def BinaryOutputFunc2(data, config, *args, **params):
    #ret = {"ss": 0, "sd": 0}
    ret = {}
    for key in data:
        ret[key] = 0
        if data[key]['all'] != 0:
            ret[key] = round(data[key]['right'] / data[key]['all'], 4)
    return json.dumps(ret)