import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

def choose_predict2(predict_d1,domain_id):
    predict_d1_cse, predict_d2_cse = [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i][0] == 0:
            predict_d1_cse.append(predict_d1[i,:])
        else:
            predict_d2_cse.append(predict_d1[i,:])
    if len(predict_d1_cse)!=0:
        predict_d1_cse = np.array(predict_d1_cse)
    if len(predict_d2_cse)!=0:
        predict_d2_cse = np.array(predict_d2_cse)
    return predict_d1_cse, predict_d2_cse

def choose_predict(predict_d1,predict_d2,domain_id):
    predict_d1_cse, predict_d2_cse = [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i][0] == 0:
            predict_d1_cse.append(predict_d1[i,:])
        else:
            predict_d2_cse.append(predict_d2[i,:])
    if len(predict_d1_cse)!=0:
        predict_d1_cse = np.array(predict_d1_cse)
    if len(predict_d2_cse)!=0:
        predict_d2_cse = np.array(predict_d2_cse)
    return predict_d1_cse, predict_d2_cse

def choose_predict_SDoverlap(predict_d1,overlap_label):
    predict_d1_cse_over, predict_d1_cse_nono = [], []
    for i in range(predict_d1.shape[0]):
        if overlap_label[i][0]==0:
            predict_d1_cse_nono.append(predict_d1[i,:])
        else:
            predict_d1_cse_over.append(predict_d1[i,:])
    if len(predict_d1_cse_over)!=0:
        predict_d1_cse_over = np.array(predict_d1_cse_over)
    if len(predict_d1_cse_nono)!=0:
        predict_d1_cse_nono = np.array(predict_d1_cse_nono)
    return predict_d1_cse_over, predict_d1_cse_nono

def choose_predict_overlap(predict_d1,predict_d2,domain_id,overlap_label):
    predict_d1_cse_over, predict_d1_cse_nono, predict_d2_cse_over, predict_d2_cse_nono = [], [], [], []
    for i in range(domain_id.shape[0]):
        if domain_id[i][0] == 0:
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

def cal_loss_cl_all(u_feat_m1_d1, u_feat_m1_d2, u_feat_m2_d1,u_feat_m2_d2, u_feat_m3_d1,u_feat_m3_d2, u_feat_m4_d1, u_feat_m4_d2):
    user_m1_query, user_m1_key = [], []
    user_m1_query.append(u_feat_m1_d1)
    user_m1_key.append(u_feat_m1_d2)
    user_m1_key.append(u_feat_m2_d1)
    user_m1_key.append(u_feat_m2_d2)
    user_m1_key.append(u_feat_m3_d1)
    user_m1_key.append(u_feat_m3_d2)
    user_m1_key.append(u_feat_m4_d1)
    user_m1_key.append(u_feat_m4_d2)
    user_m1_key = torch.stack(user_m1_key,dim=-1).squeeze()
    user_m1_query = torch.stack(user_m1_query,dim=-1).squeeze()
    user_m1_query = torch.transpose(user_m1_query.unsqueeze(-1),1,2)
    logits_m1 = torch.matmul(user_m1_query,user_m1_key).squeeze()

    user_m2_query, user_m2_key = [], []
    user_m2_query.append(u_feat_m2_d1)
    user_m2_key.append(u_feat_m2_d2)
    user_m2_key.append(u_feat_m1_d1)
    user_m2_key.append(u_feat_m1_d2)
    user_m2_key.append(u_feat_m3_d1)
    user_m2_key.append(u_feat_m3_d2)
    user_m2_key.append(u_feat_m4_d1)
    user_m2_key.append(u_feat_m4_d2)
    user_m2_key = torch.stack(user_m2_key,dim=-1).squeeze()
    user_m2_query = torch.stack(user_m2_query,dim=-1).squeeze()
    user_m2_query = torch.transpose(user_m2_query.unsqueeze(-1),1,2)
    logits_m2 = torch.matmul(user_m2_query,user_m2_key).squeeze()

    user_m3_query, user_m3_key = [], []
    user_m3_query.append(u_feat_m3_d1)
    user_m3_key.append(u_feat_m3_d2)
    user_m3_key.append(u_feat_m1_d1)
    user_m3_key.append(u_feat_m1_d2)
    user_m3_key.append(u_feat_m2_d1)
    user_m3_key.append(u_feat_m2_d2)
    user_m3_key.append(u_feat_m4_d1)
    user_m3_key.append(u_feat_m4_d2)
    user_m3_key = torch.stack(user_m3_key,dim=-1).squeeze()
    user_m3_query = torch.stack(user_m3_query,dim=-1).squeeze()
    user_m3_query = torch.transpose(user_m3_query.unsqueeze(-1),1,2)
    logits_m3 = torch.matmul(user_m3_query,user_m3_key).squeeze()

    user_m4_query, user_m4_key = [], []
    user_m4_query.append(u_feat_m4_d1)
    user_m4_key.append(u_feat_m4_d2)
    user_m4_key.append(u_feat_m1_d1)
    user_m4_key.append(u_feat_m1_d2)
    user_m4_key.append(u_feat_m2_d1)
    user_m4_key.append(u_feat_m2_d2)
    user_m4_key.append(u_feat_m3_d1)
    user_m4_key.append(u_feat_m3_d2)
    user_m4_key = torch.stack(user_m4_key,dim=-1).squeeze()
    user_m4_query = torch.stack(user_m4_query,dim=-1).squeeze()
    user_m4_query = torch.transpose(user_m4_query.unsqueeze(-1),1,2)
    logits_m4 = torch.matmul(user_m4_query,user_m4_key).squeeze()
    # print(user_spf3_query.shape,user_spf3_key.shape) #torch.Size([1024, 1, 128]) torch.Size([1024, 128, 7])
    # u_feat_enhance_m1_d1 = u_feat_enhance_m1_d1.unsqueeze(-1)
    # u_feat_enhance_m1_d2 = u_feat_enhance_m1_d2.unsqueeze(-1).repeat(1,1,u_feat_enhance_m1_d2.shape[0])
    # u_feat_enhance_m1_d2 = torch.transpose(u_feat_enhance_m1_d2,1,2)
    # logit_cl_m1 = torch.matmul(u_feat_enhance_m1_d2,u_feat_enhance_m1_d1).squeeze()
    labels = torch.LongTensor(torch.zeros(logits_m1.shape[0]).long()).cuda()
    # for i in range(logit_cl_m1.shape[0]):
    #     labels[i] = i
    loss_cl = nn.CrossEntropyLoss()(logits_m1,labels)+nn.CrossEntropyLoss()(logits_m2,labels)+nn.CrossEntropyLoss()(logits_m3,labels)+nn.CrossEntropyLoss()(logits_m4,labels)
    return loss_cl

def cal_loss_cl_refine(u_feat_enhance_m3_d1,u_feat_enhance_m4_d1):
    u_feat_enhance_m3_d1 = F.normalize(u_feat_enhance_m3_d1, dim=-1)
    u_feat_enhance_m4_d1 = F.normalize(u_feat_enhance_m4_d1, dim=-1)
    logit_cl = torch.matmul(u_feat_enhance_m3_d1,u_feat_enhance_m4_d1.T)
    # norm_num = 10 ** (len(str(int(logit_cl[0][0].item()))))
    # logit_cl = logit_cl / norm_num
    # print("logit_cl init:{}".format(logit_cl))
    # logit_cl = F.normalize(logit_cl, p=1, dim=1)
    logit_cl = torch.exp(logit_cl/0.07)#/0.07
    # print("logit_cl:{}".format(logit_cl))
    pos_logit = torch.diag(logit_cl)
    neg_logit = logit_cl.sum(dim=1)
    # print(neg_logit)
    loss_cl = -torch.log(pos_logit/neg_logit)
    return loss_cl.mean()

def cal_loss_cl(u_feat_enhance_m1_d1,u_feat_enhance_m1_d2):
    user_m1_query = []
    user_m1_query.append(u_feat_enhance_m1_d1)
    # user_m1_key.append(u_feat_enhance_m1_d2)
    u_feat_enhance_m1_d2_neg = u_feat_enhance_m1_d2.unsqueeze(-1).repeat(1,1,u_feat_enhance_m1_d2.shape[0])
    for i in range(u_feat_enhance_m1_d2.shape[0]):
        u_feat_enhance_m1_d2_neg[i,:,i] = 0
    # user_m1_key.append(u_feat_enhance_m1_d2_neg)
    # user_m1_key = torch.stack(user_m1_key,dim=-1).squeeze()
    u_feat_enhance_m1_d2 = u_feat_enhance_m1_d2.unsqueeze(-1)
    user_m1_key = torch.cat((u_feat_enhance_m1_d2,u_feat_enhance_m1_d2_neg),-1)
    user_m1_query = torch.stack(user_m1_query,dim=-1).squeeze()
    user_m1_query = torch.transpose(user_m1_query.unsqueeze(-1),1,2)
    logits_m1 = torch.matmul(user_m1_query,user_m1_key).squeeze()
    loss_cl = nn.CrossEntropyLoss()(logits_m1,labels)
    return loss_cl

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def concat_emb_listsV1(dspf_embs, dsad_embs):
    pos_embs, neg_embs = list(),list()
    for i in range(len(dspf_embs)):
        for j in range(len(dspf_embs[i])):
            pos_embs.append(dspf_embs[i][j].squeeze())
            neg_embs.append(dsad_embs[i][0].squeeze())
            for k in range(len(dspf_embs[i])):
                if k != j:
                    pos_embs.append(dspf_embs[i][j].squeeze())
                    neg_embs.append(dspf_embs[i][k].squeeze())
    pos_embs = torch.cat(pos_embs,0)
    neg_embs = torch.cat(neg_embs,0)
    label = torch.Tensor(torch.zeros(pos_embs.shape[0])).cuda()
    return pos_embs, neg_embs, label

def concat_emb_listsV0(mf_embs_compare,ml_embs_compare):
    mf_embs,ml_embs = list(),list()
    for tmp in mf_embs_compare:
        tmp = torch.cat(tmp,dim=1).squeeze()
        mf_embs.append(tmp)
    for tmp in ml_embs_compare:
        tmp = torch.cat(tmp,dim=1).squeeze()
        ml_embs.append(tmp)
    mf_embs = torch.cat(mf_embs,dim=0)
    ml_embs = torch.cat(ml_embs,dim=0)
    label = torch.Tensor(torch.zeros(mf_embs.shape[0])).cuda()
    return mf_embs, ml_embs, label

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        print(dist_sq)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

def split_domain(predict,labels,domain_ids):
    p_d1, l_d1, p_d2, l_d2, p_d3, l_d3 = list(), list(), list(), list(), list(), list()
    for i in range(predict.shape[0]):
        if domain_ids[i] == 0:
            p_d1.append(predict[i].item())
            l_d1.append(labels[i].item())
        elif domain_ids[i] == 1:
            p_d2.append(predict[i].item())
            l_d2.append(labels[i].item())
        elif domain_ids[i] == 2:
            p_d3.append(predict[i].item())
            l_d3.append(labels[i].item())
        else:
            print("error in domain id\n")
    return p_d1, l_d1, p_d2, l_d2, p_d3, l_d3#np.array(p_d1), np.array(l_d1), np.array(p_d2), np.array(l_d2), np.array(p_d3), np.array(l_d3)

class AverageMeter(object):
    def __init__(self, *keys: str):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self._check_attr(key)
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr: str) -> float:
        self._check_attr(attr)
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0

    def _check_attr(self, attr: str) -> None:
        assert attr in self.totals and attr in self.counts

def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)

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