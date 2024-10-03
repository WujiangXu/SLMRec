import numpy as np
import random
import os
import torch
# import pickle
import time
from collections import defaultdict
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
import torch.nn.functional as F
#from model import *
from utils.data_utils import *
from utils.eval_utils import *
from model import *
from sklearn.metrics import roc_auc_score
from pathlib import Path
# from pypai.model import upload_model
from tqdm import tqdm
from functools import partial
import logging
from utils import *
from utils_old import *
# from thop import profile
from sklearn.model_selection import train_test_split  # 划分数据集
from torch.utils.data import DataLoader, RandomSampler

logger = logging.getLogger()

def test(model,args,valLoader):
    model.eval()
    # stats = AverageMeter('loss')
    stats = AverageMeter('loss','ndcg_1','ndcg_5','ndcg_10','hit_1','hit_5','hit_10','MRR')
    pred_list = None
    answer_list = None
    for k,sample in enumerate(tqdm(valLoader)):
        batch = tuple(t for t in sample)
        user_ids, input_ids, target_pos, target_neg, answers, neg_samples, _ = batch
        input_ids = input_ids.cuda()
        answers = answers.cuda()
        neg_samples = neg_samples.cuda()
        with torch.no_grad():
            pos_logits, neg_logits = model.predict_sample(input_ids, answers, neg_samples)
        pos_label = torch.ones_like(pos_logits).cuda()
        neg_label = torch.zeros_like(neg_logits).cuda()
        loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
        loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
        loss = loss_real + loss_false
        predict = torch.cat((pos_logits,neg_logits),-1).squeeze().cpu().detach().numpy().copy()
        pos_logits = pos_logits.squeeze()
        neg_logits = torch.mean(neg_logits.squeeze(),-1)
        
        HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = get_sample_scores(predict)
        stats.update(loss=loss.item(),ndcg_1=NDCG_1,ndcg_5=NDCG_5,ndcg_10=NDCG_10,hit_1=HIT_1,hit_5=HIT_5,hit_10=HIT_10,MRR=MRR)
    return stats.loss, stats.hit_1, stats.ndcg_1, stats.hit_5, stats.ndcg_5, stats.hit_10, stats.ndcg_10, stats.MRR

def train(model,device,trainLoader,args,valLoader,testLoader):
    best_mrr = 0

    save_path = Path(args.model_dir) / 'checkpoint' / 'best.pt'
    criterion_cls = nn.BCELoss(reduce=False)

    if not os.path.exists(os.path.join(Path(args.model_dir),'checkpoint')):
        os.mkdir(os.path.join(Path(args.model_dir),'checkpoint'))

    for epoch in range(args.epoch):
        stats = AverageMeter('train_loss','val_loss','test_loss')
        model.train()
        for i,sample in enumerate(tqdm(trainLoader)):

            batch = tuple(t for t in sample)
            user_ids, input_ids, target_pos, target_neg, answers, neg_samples, _ = batch
            # print(target_pos.shape,target_neg.shape,answers.shape)
            input_ids = input_ids.cuda()
            answers = answers.cuda()
            neg_samples = neg_samples.cuda()
            loss = model(input_ids,answers)

            # large industry dataset
            # pos_logits, neg_logits = model.predict_sample(input_ids, answers, neg_samples)
            # pos_label = torch.ones_like(pos_logits).cuda()
            # neg_label = torch.zeros_like(neg_logits).cuda()
            # loss_real = nn.BCEWithLogitsLoss()(pos_logits, pos_label)
            # loss_false = nn.BCEWithLogitsLoss()(neg_logits, neg_label)
            # loss = loss_real + loss_false
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            stats.update(train_loss=loss.item())
            if i % 20 == 0:
                logger.info(f'train total loss:{stats.train_loss} \t')
            #print("epoch :{} train loss:{}, auc:{}".format(epoch,stats.loss,stats.auc)) 
        
        val_loss, HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR = test(model,args,valLoader)
        logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                    f'val loss: {val_loss:.4f}\t'
                    f'HR@1: {HIT_1:.4f}\t,' 
                    f'HR@5: {HIT_5:.4f}\t, '
                    f'HR@10: {HIT_10:.4f} \t'
                    f'NDCG@1: {NDCG_1:.4f}\t,' 
                    f'NDCG@5: {NDCG_5:.4f}\t, '
                    f'NDCG@10: {NDCG_10:.4f} \t'
                    f'MRR_test: {MRR:.4f} \t')
        if MRR > best_mrr:
            torch.save(model.state_dict(), str(save_path))
            # test 
            test_loss, HIT_1_test, NDCG_1_test, HIT_5_test, NDCG_5_test, HIT_10_test, NDCG_10_test, MRR_test = test(model,args,testLoader)
            logger.info(f'Epoch: {epoch}/{args.epoch} \t'
                    f'test loss: {test_loss:.4f}\t'
                    f'HR@1: {HIT_1_test:.4f}\t,' 
                    f'HR@5: {HIT_5_test:.4f}\t, '
                    f'HR@10: {HIT_10_test:.4f} \t'
                    f'NDCG@1: {NDCG_1_test:.4f}\t,' 
                    f'NDCG@5: {NDCG_5_test:.4f}\t, '
                    f'NDCG@10: {NDCG_10_test:.4f} \t'
                    f'MRR_test: {MRR_test:.4f} \t')
        best_mrr = max(best_mrr,MRR)

    return HIT_1_test, NDCG_1_test, HIT_5_test, NDCG_5_test, HIT_10_test, NDCG_10_test, MRR_test

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SR training')
    parser.add_argument('--epoch', type=int, default=50, help='# of epoch')
    parser.add_argument('--bs', type=int, default=1024, help='# images in batch')
    parser.add_argument('--use_gpu', type=bool, default=True, help='gpu flag, true for GPU and false for CPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden layer dim')
    parser.add_argument('-sl','--max_seq_length', type=int, default=30, help='the length of the sequence')
    parser.add_argument('--item_size', type=int, default=999, help='sample negative numbers')
    parser.add_argument('--overlap_ratio', type=float, default=0.5, help='overlap ratio for choose dataset ')
    parser.add_argument('--layers', type=int, default=2, help='stacked sasrec')
    parser.add_argument('-md','--model-dir', type=str, default='model/')
    parser.add_argument('--log-file', type=str, default='log')
    parser.add_argument('--model', type=str, default='sasrec', help='model select')
    parser.add_argument('-ds','--dataset_type', type=str, default='amazon')
    parser.add_argument('-dm','--domain_type', type=str, default='cloths')


    args = parser.parse_args()

    # train val best for test
    # predict mode: sample_num/all 

    # for i in range(1):
    SEED = 9999 #  a set value
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.log_file = "log" + ".txt"
    
    datasetTrain = SASRecDataset(item_size=args.item_size, max_seq_length=args.max_seq_length,data_type='train',csv_path="./dataset/sequential/{}.csv".format(args.domain_type))
    trainLoader = data.DataLoader(datasetTrain, batch_size=args.bs, shuffle=True, num_workers=8)

    datasetVal = SASRecDataset(item_size=args.item_size, max_seq_length=args.max_seq_length,data_type='valid',csv_path="./dataset/sequential/{}.csv".format(args.domain_type))
    valLoader = data.DataLoader(datasetVal, batch_size=args.bs, shuffle=False, num_workers=8)

    datasetTest = SASRecDataset(item_size=args.item_size, max_seq_length=args.max_seq_length,data_type='test',csv_path="./dataset/sequential/{}.csv".format(args.domain_type))
    testLoader = data.DataLoader(datasetTest, batch_size=args.bs, shuffle=False, num_workers=8)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SASRec(args,device,datasetTest).cuda()
    # elif args.model.lower() == "bert4rec":
    #     model = BERT4Rec(user_length=user_length, user_emb_dim=args.emb_dim, item_length=item_length, item_emb_dim=args.emb_dim, seq_len=args.seq_len, hid_dim=args.hid_dim, bs=args.bs, isInC=args.isInC, isItC=args.isItC, threshold1=args.ts1, threshold2=args.ts2).cuda()
    print("find cuda right !!\n")
    # if cuda:
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')

    if cuda:
        #model = torch.nn.DataParallel(model)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #cudnn.benchmark = True
        model = model.cuda()
        # model.to(device)
        print("use cuda!")
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    init_logger(args.model_dir, args.log_file)
    logger.info(vars(args))
    # if os.path.exists(args.model_dir + "best_d1.pt"):
    #     print("load_pretrained")
    #     state_dict = torch.load(args.model_dir + "best_d1.pt")
    #     model.load_state_dict(state_dict,strict=False)
    train(model,device,trainLoader,args,valLoader,testLoader)
    # test(model,args,testLoader)