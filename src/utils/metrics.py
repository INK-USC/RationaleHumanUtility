import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import precision_recall_curve, auc, f1_score, average_precision_score


def init_best_metrics():
    return {
        'best_epoch': 0,
        'dev_best_perf': None,
        'test_best_perf': None,
    }

def init_perf_metrics():
    # assert num_classes >= 2
    perf_metrics = torch.nn.ModuleDict({
        'acc': torchmetrics.Accuracy(),
    })
    return perf_metrics

def calc_preds(logits):
    return torch.argmax(logits, dim=1)

def get_step_metrics(preds, labels, metrics):
    res = {}
    for key, metric_fn in metrics.items():
        res.update({key: metric_fn(preds, labels) * 100})
    return res

def get_epoch_metrics(metrics):
    res = {}
    for key, metric_fn in metrics.items():
        res.update({key: metric_fn.compute() * 100})
        metric_fn.reset()
    return res

def process_outputs(outputs, tokenizer, is_infilling=False):
    labels, rationales = [], []
    if is_infilling:
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    else:
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(decoded_outputs)
    for text in decoded_outputs:
        # print(text)
        if "<extra_id_0>" in text and '<extra_id_1>' in text :            
            splitted_text = text.split("<extra_id_1>")
            if "<extra_id_2>" not in text:
                pred_e=''
            else:
                pred_e = splitted_text[1].split("<extra_id_2>")[0].strip()
            pred_l = splitted_text[0].split("<extra_id_0>")[-1].strip()
            labels.append(pred_l)
            rationales.append(pred_e)
        elif "because" in text:
            splitted_text = text.split("because")
            # print(splitted_text)
            pred_l = splitted_text[0].strip()
            pred_e = splitted_text[1].strip()
            labels.append(pred_l)
            rationales.append(pred_e)
        
        else:
            
            splitted_text = text.split(" ")
            if len(splitted_text)==1:
                pred_e=''
                pred_l=splitted_text[0].strip()
            else:
                pred_l = splitted_text[0].strip()
                pred_e = splitted_text[1].strip()
            labels.append(pred_l)
            rationales.append(pred_e)
        # else:
        #     raise NotImplementedError
    # Need to figure this out for I-O cases
    return labels, rationales

# def process_outputs(outputs, tokenizer):
#     labels, rationales = [], []
#     decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     for text in decoded_outputs:
#         text_split = text.split('explanation:')
        
#         pred_l = text_split[0].strip()
#         if len(text_split) > 1:
#             pred_e = text_split[1].strip()
#             pred_e = pred_e.split('<extra_id')[0].strip()
#         else:
#             pred_e = ''

#         labels.append(pred_l)
#         rationales.append(pred_e)

#     return labels, rationales
