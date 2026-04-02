from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

def multi_class_evaluation(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,{image ids:[]}  {0: [torch.tensor(0)], 1: [torch.tensor(5)],...}
    :param res: Dictionary with the image ids ant their generated captions
    #{0: [torch.tensor([ 0.1250, -0.1372, -0.6413, -1.3435, -0.9643, -0.6389, -1.0454, -0.6111, -0.7066, -1.5715, -0.0803])],\
    # 1: [torch.tensor([-0.4086, -0.7441, -1.1091, -1.3233, -1.1271,  2.1074, -1.7877, -1.3648, -1.9576, -2.9300, -2.0747])]}
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    
    gts1 = []
    for v in gts.values():
        gts1.append(v[0].item())

    res1 = []
    for v in res.values():
        res1.append(torch.max(v[0],0)[1].cpu().numpy())

    eval_res = {}
    # Compute score for each metric
    eval_res['accuracy'] = round(accuracy_score(gts1, res1),4)    
    eval_res['precision'] = round(precision_score(gts1, res1, average='macro'),4)
    eval_res['recall'] = round(recall_score(gts1, res1, average='macro'),4)
    eval_res['f1_score'] = round(f1_score(gts1, res1, average='macro'),4)
    return eval_res



def multi_label_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),  # subset accuracy
        "f1_score": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "micro_f1": round(f1_score(y_true, y_pred, average='micro', zero_division=0), 4)
    }
