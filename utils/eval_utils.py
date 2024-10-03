import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn

# ====================Metrics==============================
def RecallPrecision_atK(test, r, k):
    tp = r[:, :k].sum(1)
    precision = np.sum(tp) / k
    recall_n = np.array([len(test[i]) for i in range(len(test))])
    recall = np.sum(tp / recall_n)
    return precision, recall


def MRR_atK(test, r, k):
    pred = r[:, :k]
    weight = np.arange(1, k+1)
    MRR = np.sum(pred / weight, axis=1) / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MRR = np.sum(MRR)
    return MRR


def MAP_atK(test, r, k):
    pred = r[:, :k]
    rank = pred.copy()
    for i in range(k):
        rank[:, k - i - 1] = np.sum(rank[:, :k - i], axis=1)
    weight = np.arange(1, k+1)
    AP = np.sum(pred * rank / weight, axis=1)
    AP = AP / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MAP = np.sum(AP)
    return MAP


def NDCG_atK(test, r, k):
    pred = r[:, :k]
    test_mat = np.zeros((len(pred), k))
    for i, items in enumerate(test):
        length = k if k <= len(items) else len(items)
        test_mat[i, :length] = 1

    idcg = np.sum(test_mat * (1. / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    dcg = pred * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    ndcg = np.sum(ndcg)
    return ndcg


def AUC(all_item_scores, dataset, test):
    r_all = np.zeros((dataset.m_item, ))
    r_all[test] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test, pred):
    r = []
    for i in range(len(test)):
        groundTruth, predTopK = test[i], pred[i]
        hits = list(map(lambda x: x in groundTruth, predTopK))
        hits = np.array(hits).astype("float")
        r.append(hits)
    return np.array(r).astype('float')
# ====================end Metrics=============================
def get_sample_scores(pred_list):
    pred_list = (-pred_list).argsort().argsort()[:, 0]
    HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
    HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
    HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
    return HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def choose_predict(predict_d1,predict_d2,domain_id):
    predict_d1_cse, predict_d2_cse = [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i] == 0:
            predict_d1_cse.append(predict_d1[i,:])
        else:
            predict_d2_cse.append(predict_d2[i,:])
    if len(predict_d1_cse)!=0:
        predict_d1_cse = np.array(predict_d1_cse)
    if len(predict_d2_cse)!=0:
        predict_d2_cse = np.array(predict_d2_cse)
    return predict_d1_cse, predict_d2_cse

def choose_predict_overlap(predict_d1,predict_d2,domain_id,overlap_label):
    predict_d1_cse_over, predict_d1_cse_nono, predict_d2_cse_over, predict_d2_cse_nono = [], [], [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i] == 0:
            if overlap_label[i][0]==0:
                predict_d1_cse_nono.append(predict_d1[i,:])
            else:
                predict_d1_cse_over.append(predict_d1[i,:])
        else:
            if overlap_label[i][0]==0:
                predict_d2_cse_nono.append(predict_d2[i,:])
            else:
                predict_d2_cse_over.append(predict_d2[i,:])
    if len(predict_d1_cse_over)!=0:
        predict_d1_cse_over = np.array(predict_d1_cse_over)
    if len(predict_d1_cse_nono)!=0:
        predict_d1_cse_nono = np.array(predict_d1_cse_nono)
    if len(predict_d2_cse_over)!=0:
        predict_d2_cse_over = np.array(predict_d2_cse_over)
    if len(predict_d2_cse_nono)!=0:
        predict_d2_cse_nono = np.array(predict_d2_cse_nono)
    return predict_d1_cse_over, predict_d1_cse_nono, predict_d2_cse_over, predict_d2_cse_nono

def compute_metrics(pred):
    logits = pred.predictions 
    # print("logits shape:{}".format(logits.shape))
    if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
        HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = -1, -1, -1, -1, -1, -1, -1
    else:
        HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = get_sample_scores(logits)
    return {
        'hit@1':HIT_1,
        'hit@5':HIT_5,
        'ndcg@5':NDCG_5,
        'hit@10':HIT_10,
        'ndcg@10':NDCG_10,
        'mrr':MRR,
    }

def compute_metrics_multiple(pred):
    logits = pred.predictions 
    print(logits.shape)
    loss = logits[:,:,-1]
    predict = logits[:,:,:-1]
    # if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
    #     HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = -1, -1, -1, -1, -1, -1, -1
    # else:
    HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = [],[],[],[],[],[],[]
    for i in range(predict.shape[1]):
        HIT_1_tmp, NDCG_1_tmp, HIT_5_tmp, NDCG_5_tmp, HIT_10_tmp, NDCG_10_tmp, MRR_tmp = get_sample_scores(predict[:,i,:])
        HIT_1.append(HIT_1_tmp)
        NDCG_1.append(NDCG_1_tmp)
        HIT_5.append(HIT_5_tmp)
        NDCG_5.append(NDCG_5_tmp)
        HIT_10.append(HIT_10_tmp)
        NDCG_10.append(NDCG_10_tmp)
        MRR.append(MRR_tmp)
    print("mrr:{}".format(MRR))
    return {
        'hit@1':HIT_1,
        'hit@5':HIT_5,
        'ndcg@5':NDCG_5,
        'hit@10':HIT_10,
        'ndcg@10':NDCG_10,
        'mrr':MRR,
        'loss':np.mean(loss,axis=0),
    }
# def compute_metrics(pred):
#     logits = pred.predictions 
#     # print(pred.label_ids[0].shape,pred.label_ids[1].shape)
#     labels = pred.label_ids[0]
#     domain_id = pred.inputs # (19045)
#     logits_d1, logits_d2, cs_label, tailed_label = logits[:,:logits.shape[1]//4], logits[:,logits.shape[1]//4:logits.shape[1]//2], logits[:,logits.shape[1]//2:logits.shape[1]*3//4], logits[:,logits.shape[1]*3//4:]
#     print(logits_d1.shape, logits_d2.shape, cs_label.shape, tailed_label.shape)
#     predict_d1_cse, predict_d2_cse = choose_predict(logits_d1, logits_d2, domain_id)
#     predict_d1_cse_cs, _3, predict_d2_cse_cs, _4 = choose_predict_overlap(logits_d1,logits_d2,domain_id,cs_label)
#     predict_d1_cse_tail, _5, predict_d2_cse_tail, _6 = choose_predict_overlap(logits_d1,logits_d2,domain_id,tailed_label)
#     criterion_cls = nn.BCEWithLogitsLoss(reduce=False)
#     domain_id = torch.LongTensor(domain_id)
#     one_value = torch.LongTensor(torch.ones(domain_id.shape[0]).long())
#     mask_d1 = torch.LongTensor((one_value - domain_id).long())
#     mask_d2 = torch.LongTensor(domain_id)
#     # print(logits_d1.shape,labels.shape)
#     val_loss_d1 = torch.mean(criterion_cls(torch.FloatTensor(logits_d1),torch.FloatTensor(labels)) * mask_d1.unsqueeze(1))
#     val_loss_d2 = torch.mean(criterion_cls(torch.FloatTensor(logits_d2),torch.FloatTensor(labels)) * mask_d2.unsqueeze(1))
#     # val_loss
#     HIT_1_d1, NDCG_1_d1, HIT_5_d1, NDCG_5_d1, HIT_10_d1, NDCG_10_d1, MRR_d1 = get_sample_scores(predict_d1_cse)
#     HIT_1_d2, NDCG_1_d2, HIT_5_d2, NDCG_5_d2, HIT_10_d2, NDCG_10_d2, MRR_d2 = get_sample_scores(predict_d2_cse)
#     HIT_1_d1_cs, NDCG_1_d1_cs, HIT_5_d1_cs, NDCG_5_d1_cs, HIT_10_d1_cs, NDCG_10_d1_cs, MRR_d1_cs = get_sample_scores(predict_d1_cse_cs)
#     HIT_1_d2_cs, NDCG_1_d2_cs, HIT_5_d2_cs, NDCG_5_d2_cs, HIT_10_d2_cs, NDCG_10_d2_cs, MRR_d2_cs = get_sample_scores(predict_d2_cse_cs)

#     HIT_1_d1_tailed, NDCG_1_d1_tailed, HIT_5_d1_tailed, NDCG_5_d1_tailed, HIT_10_d1_tailed, NDCG_10_d1_tailed, MRR_d1_tailed = get_sample_scores(predict_d1_cse_tail)
#     HIT_1_d2_tailed, NDCG_1_d2_tailed, HIT_5_d2_tailed, NDCG_5_d2_tailed, HIT_10_d2_tailed, NDCG_10_d2_tailed, MRR_d2_tailed = get_sample_scores(predict_d2_cse_tail)
#     return {
#         'val_loss_d1':val_loss_d1,
#         'val_loss_d2':val_loss_d2,
#         'd1_hit@1':HIT_1_d1,
#         'd1_hit@5':HIT_5_d1,
#         'd1_ndcg@5':NDCG_5_d1,
#         'd1_hit@10':HIT_10_d1,
#         'd1_ndcg@10':NDCG_10_d1,
#         'd1_mrr':MRR_d1,
#         'd2_hit@1':HIT_1_d2,
#         'd2_hit@5':HIT_5_d2,
#         'd2_ndcg@5':NDCG_5_d2,
#         'd2_hit@10':HIT_10_d2,
#         'd2_ndcg@10':NDCG_10_d2,
#         'd2_mrr':MRR_d2,
#         'd1_hit@1_cs':HIT_1_d1_cs,
#         'd1_hit@5_cs':HIT_5_d1_cs,
#         'd1_ndcg@5_cs':NDCG_5_d1_cs,
#         'd1_hit@10_cs':HIT_10_d1_cs,
#         'd1_ndcg@10_cs':NDCG_10_d1_cs,
#         'd1_mrr_cs':MRR_d1_cs,
#         'd2_hit@1_cs':HIT_1_d2_cs,
#         'd2_hit@5_cs':HIT_5_d2_cs,
#         'd2_ndcg@5_cs':NDCG_5_d2_cs,
#         'd2_hit@10_cs':HIT_10_d2_cs,
#         'd2_ndcg@10_cs':NDCG_10_d2_cs,
#         'd2_mrr_cs':MRR_d2_cs,
#         'd1_hit@1_tailed':HIT_1_d1_tailed,
#         'd1_hit@5_tailed':HIT_5_d1_tailed,
#         'd1_ndcg@5_tailed':NDCG_5_d1_tailed,
#         'd1_hit@10_tailed':HIT_10_d1_tailed,
#         'd1_ndcg@10_tailed':NDCG_10_d1_tailed,
#         'd1_mrr_tailed':MRR_d1_tailed,
#         'd2_hit@1_tailed':HIT_1_d2_tailed,
#         'd2_hit@5_tailed':HIT_5_d2_tailed,
#         'd2_ndcg@5_tailed':NDCG_5_d2_tailed,
#         'd2_hit@10_tailed':HIT_10_d2_tailed,
#         'd2_ndcg@10_tailed':NDCG_10_d2_tailed,
#         'd2_mrr_tailed':MRR_d2_tailed,
#     }


def get_full_sort_score(answers, pred_list):
    recall, ndcg, mrr = [], [], []
    for k in [5, 10, 15, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
        mrr.append(MRR_atK(answers, pred_list, k))
    return recall, ndcg, mrr