from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np


def eval_AUPR(labels, scores):
    return average_precision_score(labels, scores)

def eval_AUROC(labels, scores):
    return roc_auc_score(labels, scores)

def all_eval(labels, scores):
    AUPR = eval_AUPR(labels, scores)
    print('Test Dataset AUPR:', AUPR)

    AUROC = eval_AUROC(labels, scores)
    print('Test Dataset AUROC:', AUROC)
    
    return [f'AUPR:{AUPR}', f'AUROC:{AUROC}']