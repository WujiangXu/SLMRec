import pandas as pd
import numpy as np
import torch
import time
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import json
import random
import spacy
import ast

def parse_list(string):
    try:
        # 将NaN值替换为None
        string = string.replace('nan', "'None'")
        # 安全地将字符串解析为原始列表
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        # 如果解析失败，返回一个空列表或适当的默认值
        return []

class SRDatasetText(Dataset):
    def __init__(self,seq_len,llama_tokenizer,data_type,neg_nums,long_length,pad_id,csv_path=''):
        super(SRDatasetText, self).__init__()
        self.df = pd.read_csv(csv_path)
        self.df['title'] = self.df['title'].apply(parse_list)
        self.reviewerID = self.df['reviewerID'].tolist()#self.df['reviewerID'].tolist()
        self.llama_tokenizer = llama_tokenizer
        self.asin = self.df['asin'].tolist()#self.df['asin_x'].tolist()
        self.title = self.df['title'].tolist()
        self.tail_len = self.__return_tailed_length__(self.df['asin'])
        self.item_pool = self.__build_i_set__(self.asin)     
        self.seq_len = seq_len
        self.data_type = data_type # train val test 
        assert self.data_type in {"train", "valid", "test"}
        self.neg_nums = neg_nums
        self.long_length = long_length
        self.pad_id = pad_id

    def __return_tailed_length__(self,seq):
        lengths = seq.apply(lambda x: len(json.loads(x)) if len(json.loads(x)) > 2 else None)  # 将 JSON 字符串转换为列表并计算长度，仅限列表长度大于2的情况
        lengths = lengths.dropna() 
        sorted_lengths = lengths.sort_values(ascending=False)  # 按照长度降序排序
        tail_80_percent = sorted_lengths.iloc[int(len(sorted_lengths) * 0.2):]  # 获取尾部 80% 的子列表
        average_length = tail_80_percent.mean()  # 计算子列表的平均长度
        return int(average_length)

    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = json.loads(item_seq)
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __len__(self):
        print("dataset len:{}\n".format(len(self.reviewerID)))
        return len(self.reviewerID)

    def __getitem__(self, idx):
        user_node = self.reviewerID[idx]
        seq_tmp = json.loads(self.asin[idx])
        title_tmp = self.title[idx]
        label = list()
        # neg_items_set = self.item_pool - set(seq_tmp) # neglect
        if len(seq_tmp)>self.tail_len:
            tailed_label_tmp = 0
        else:
            tailed_label_tmp = 1
        item = seq_tmp[-1]
        seq_tmp = seq_tmp[:-1]
        title_tmp = title_tmp[:-1]
        label.append(1)
        while(item in seq_tmp):
            seq_tmp.remove(item)
        # print("seq after:{}".format(seq_d1_tmp))
        if self.isTrain:
            neg_samples = random.sample(neg_items_set, 1)
            label.append(0)
        else:
            neg_samples = random.sample(neg_items_set, self.neg_nums)
            for _ in range(self.neg_nums):
                label.append(0)

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        
        seq_tmp,input_mask = seq_padding(seq_tmp,self.seq_len+1,self.pad_id)
        sample = dict()
        # sample['user_node'] = np.array([user_node])
        sample['i_node'] = np.array([item])
        sample['seq'] = np.array([seq_tmp])
        sample['input_mask'] = np.array([input_mask])
        sample['tailed_label'] = np.array([tailed_label_tmp]) # if 1 long-tailed user
        sample['label'] = np.array(label)
        sample['neg_samples'] = np.array(neg_samples)

        # text tokenize for LLM
        self.title_tmp = ' '.join(title_tmp)

        self.title_tmp_ids, self.title_tmp_mask = self.llama_tokenizer(self.title_tmp,truncation=True,max_length=500,padding='max_length',return_tensors='pt', add_special_tokens=False).values()

        sample['textids_info'] = self.title_tmp_ids#torch.cat((self.title_tmp_ids,self.reviewText_tmp_ids,self.feature_tmp_ids,self.brand_tmp_ids,self.category_tmp_ids,self.description_tmp_ids),-1)
        sample['textmask_info'] = self.title_tmp_mask#torch.cat((self.title_tmp_mask,self.reviewText_tmp_mask,self.feature_tmp_mask,self.brand_tmp_mask,self.category_tmp_mask,self.description_tmp_mask),-1)
        # print(sample.keys())
        return sample

@dataclass
class SRTextCollator:
    def __call__(self, batch) -> dict:
        # sample = zip(*batch)
        # print(batch)
        # for sample in batch:
        #     print("in batch keys:{}".format(sample.keys()))

        # user_node = torch.cat([ torch.LongTensor(sample['user_node']) for sample in batch],dim=0)
        i_node = torch.cat([ torch.LongTensor(sample['i_node']) for sample in batch],dim=0)
        seq = torch.cat([ torch.LongTensor(sample['seq']) for sample in batch],dim=0)
        input_mask = torch.cat([ torch.LongTensor(sample['input_mask']) for sample in batch],dim=0)
        labels = torch.stack([ torch.LongTensor(sample['label']) for sample in batch],dim=0)
        tailed_label = torch.cat([ torch.LongTensor(sample['tailed_label']) for sample in batch],dim=0)
        textids_info = torch.cat([ sample['textids_info'] for sample in batch],dim=0)
        textmask_info = torch.cat([ sample['textmask_info'] for sample in batch],dim=0)
        neg_samples = torch.stack([ torch.LongTensor(sample['neg_samples']) for sample in batch],dim=0)
        data = {
            # 'user_node' : user_node,
                'i_node': i_node,
                'seq' : seq,
                'input_mask' : input_mask,
                'tailed_label' : tailed_label,
                'labels':labels,
                'input_ids':i_node,
                'neg_samples':neg_samples,
                'textids_info':textids_info,
                'textmask_info':textmask_info
                }
        return data

def process_string(s):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(s)
    after_str = ""
    # Iterate over the entities and print them out
    for ent in doc.ents:
        after_str += ent.text
    return after_str

def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

class SASRecDataset(Dataset):
    # no augmentation
    def __init__(self, item_size, max_seq_length, data_type='train', csv_path=''):
        # self.args = args
        self.df = pd.read_csv(csv_path)
        self.item_size = item_size
        # self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = max_seq_length
        self.df['asin'] = self.df['asin'].apply(json.loads)

        # 现在我们有了包含真实列表的DataFrame列，我们可以找出所有数字的范围
        self.m_item = max(self.df['asin'].explode())+1
        print("item length :{}".format(self.m_item))
        # self.df['title'] = self.df['title'].apply(parse_list)
        self.reviewerID = self.df['reviewerID'].tolist()#self.df['reviewerID'].tolist()
        # self.llama_tokenizer = llama_tokenizer
        self.user_seq = self.df['asin'].tolist()#self.df['asin_x'].tolist()
        # self.title = self.df['title'].tolist()
        self.tail_len = self.__return_tailed_length__(self.df['asin'])
        self.item_pool = self.__build_i_set__(self.user_seq)     
        self.seq_len = max_seq_length

    def __return_tailed_length__(self,seq):
        lengths = seq.apply(lambda x: len(x) if len(x) > 2 else None)  # 将 JSON 字符串转换为列表并计算长度，仅限列表长度大于2的情况
        lengths = lengths.dropna() 
        sorted_lengths = lengths.sort_values(ascending=False)  # 按照长度降序排序
        tail_80_percent = sorted_lengths.iloc[int(len(sorted_lengths) * 0.2):]  # 获取尾部 80% 的子列表
        average_length = tail_80_percent.mean()  # 计算子列表的平均长度
        return int(average_length)

    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = item_seq
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [items[-3]]

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        # print(items,input_ids,target_pos,answer)
        
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.item_size))

        # print(len(target_neg))
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        neg_items_set = self.item_pool - seq_set
        neg_samples = random.sample(neg_items_set, self.item_size)

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.data_type == "train":
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_samples, dtype=torch.long),
                "train",
            )
            return cur_tensors
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_samples, dtype=torch.long),
                "test",

            )
            return cur_tensors

    def __len__(self):
        return len(self.user_seq)    

class LLMDataset(Dataset):
    # no augmentation
    def __init__(self, item_size, max_seq_length, data_type='train', csv_path=''):
        # self.args = args
        self.df = pd.read_csv(csv_path)
        self.item_size = item_size
        # self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = max_seq_length
        self.df['asin'] = self.df['asin'].apply(json.loads)

        # 现在我们有了包含真实列表的DataFrame列，我们可以找出所有数字的范围
        self.m_item = max(self.df['asin'].explode())+1

        # self.df['title'] = self.df['title'].apply(parse_list)
        self.reviewerID = self.df['reviewerID'].tolist()#self.df['reviewerID'].tolist()
        # self.llama_tokenizer = llama_tokenizer
        self.user_seq = self.df['asin'].tolist()#self.df['asin_x'].tolist()
        # self.title = self.df['title'].tolist()
        self.tail_len = self.__return_tailed_length__(self.df['asin'])
        self.item_pool = self.__build_i_set__(self.user_seq)     
        self.seq_len = max_seq_length

    def __return_tailed_length__(self,seq):
        lengths = seq.apply(lambda x: len(x) if len(x) > 2 else None)  # 将 JSON 字符串转换为列表并计算长度，仅限列表长度大于2的情况
        lengths = lengths.dropna() 
        sorted_lengths = lengths.sort_values(ascending=False)  # 按照长度降序排序
        tail_80_percent = sorted_lengths.iloc[int(len(sorted_lengths) * 0.2):]  # 获取尾部 80% 的子列表
        average_length = tail_80_percent.mean()  # 计算子列表的平均长度
        return int(average_length)

    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = item_seq
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test", "test_multiple"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [items[-3]]

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        # print(items,input_ids,target_pos,answer)
        
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.item_size))

        # print(len(target_neg))
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        neg_items_set = self.item_pool - seq_set
        neg_samples = random.sample(neg_items_set, self.item_size)

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        sample = dict()

        if self.data_type == "train":
            sample['user_id'] = np.array([user_id])
            sample['input_ids'] = np.array([input_ids])
            sample['target_pos'] = np.array([target_pos])
            sample['target_neg'] = np.array([target_neg])
            sample['answer'] = np.array([answer])
            sample['neg_samples'] = np.array([neg_samples])
            sample['data_types'] = np.array([0]) # train:0,val/test:1
            return sample

        elif self.data_type == "valid" or self.data_type == "test":
            sample['user_id'] = np.array([user_id])
            sample['input_ids'] = np.array([input_ids])
            sample['target_pos'] = np.array([target_pos])
            sample['target_neg'] = np.array([target_neg])
            sample['answer'] = np.array([answer])
            sample['neg_samples'] = np.array([neg_samples])
            sample['data_types'] = np.array([1]) # train:0,val/test:1
            return sample

        elif self.data_type == "test_multiple":
            sample['user_id'] = np.array([user_id])
            sample['input_ids'] = np.array([input_ids])
            sample['target_pos'] = np.array([target_pos])
            sample['target_neg'] = np.array([target_neg])
            sample['answer'] = np.array([answer])
            sample['neg_samples'] = np.array([neg_samples])
            sample['data_types'] = np.array([2]) # train:0,val/test:1,multiple prediction test:2
            return sample

    def __len__(self):
        return len(self.user_seq)   

class P5Dataset(Dataset):
    # no augmentation
    def __init__(self, item_size, max_seq_length, data_type='train', csv_path=''):
        # self.args = args
        self.df = pd.read_csv(csv_path)
        self.item_size = item_size
        # self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = max_seq_length
        self.df['asin'] = self.df['asin'].apply(json.loads)

        # 现在我们有了包含真实列表的DataFrame列，我们可以找出所有数字的范围
        self.m_item = max(self.df['asin'].explode())+1

        # self.df['title'] = self.df['title'].apply(parse_list)
        self.reviewerID = self.df['reviewerID'].tolist()#self.df['reviewerID'].tolist()
        # self.llama_tokenizer = llama_tokenizer
        self.user_seq = self.df['asin'].tolist()#self.df['asin_x'].tolist()
        # self.title = self.df['title'].tolist()
        self.tail_len = self.__return_tailed_length__(self.df['asin'])
        self.item_pool = self.__build_i_set__(self.user_seq)     
        self.seq_len = max_seq_length

    def __return_tailed_length__(self,seq):
        lengths = seq.apply(lambda x: len(x) if len(x) > 2 else None)  # 将 JSON 字符串转换为列表并计算长度，仅限列表长度大于2的情况
        lengths = lengths.dropna() 
        sorted_lengths = lengths.sort_values(ascending=False)  # 按照长度降序排序
        tail_80_percent = sorted_lengths.iloc[int(len(sorted_lengths) * 0.2):]  # 获取尾部 80% 的子列表
        average_length = tail_80_percent.mean()  # 计算子列表的平均长度
        return int(average_length)

    def __build_i_set__(self,seq1):
        item_d1 = list()
        for item_seq in seq1:
            item_seq_list = item_seq
            for i_tmp in item_seq_list:
                item_d1.append(i_tmp)
        item_pool_d1 = set(item_d1)
        return item_pool_d1

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test", "test_multiple"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [items[-3]]

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        # print(items,input_ids,target_pos,answer)
        
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.item_size))

        # print(len(target_neg))
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        neg_items_set = self.item_pool - seq_set
        neg_samples = random.sample(neg_items_set, self.item_size)

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        sample = dict()
        if self.data_type != "test":
            sample['input_ids'] = np.squeeze(np.array([input_ids]))
            # inputs_mask = (sample['input_ids'] > 0)
            # sample['inputs_mask'] = np.squeeze(np.array([inputs_mask]))
            sample['labels'] = np.squeeze(np.array([target_pos]))
        else:
            sample['input_ids'] = np.squeeze(np.array([input_ids]))
            sample['labels'] = np.squeeze(np.array([target_pos]))
        # sample['input_ids'] = sample['inputs']
        # print(" sample['labels'] :{}".format(sample['labels']))
        # print("sample['inputs'] shape:{}, mask shape:{}, labels shape:{}".format(sample['inputs'].shape,sample['inputs_mask'].shape,sample['labels'].shape))
        return sample 
        # if self.data_type == "train":
        #     sample['user_id'] = np.array([user_id])
        #     sample['input_ids'] = np.array([input_ids])
        #     sample['target_pos'] = np.array([target_pos])
        #     sample['target_neg'] = np.array([target_neg])
        #     sample['answer'] = np.array([answer])
        #     sample['neg_samples'] = np.array([neg_samples])
        #     sample['data_types'] = np.array([0]) # train:0,val/test:1
        #     return sample

        # elif self.data_type == "valid" or self.data_type == "test":
        #     sample['user_id'] = np.array([user_id])
        #     sample['input_ids'] = np.array([input_ids])
        #     sample['target_pos'] = np.array([target_pos])
        #     sample['target_neg'] = np.array([target_neg])
        #     sample['answer'] = np.array([answer])
        #     sample['neg_samples'] = np.array([neg_samples])
        #     sample['data_types'] = np.array([1]) # train:0,val/test:1
        #     return sample

        # elif self.data_type == "test_multiple":
        #     sample['user_id'] = np.array([user_id])
        #     sample['input_ids'] = np.array([input_ids])
        #     sample['target_pos'] = np.array([target_pos])
        #     sample['target_neg'] = np.array([target_neg])
        #     sample['answer'] = np.array([answer])
        #     sample['neg_samples'] = np.array([neg_samples])
        #     sample['data_types'] = np.array([2]) # train:0,val/test:1,multiple prediction test:2
        #     return sample

    def __len__(self):
        return len(self.user_seq)   

@dataclass
class SequentialCollator:
    def __call__(self, batch) -> dict:
        user_id = torch.cat([ torch.LongTensor(sample['user_id']) for sample in batch],dim=0)
        inputs = torch.cat([ torch.LongTensor(sample['input_ids']) for sample in batch],dim=0)
        answers = torch.cat([ torch.LongTensor(sample['answer']) for sample in batch],dim=0)
        neg_samples = torch.cat([ torch.LongTensor(sample['neg_samples']) for sample in batch],dim=0)
        inputs_mask = (inputs > 0)
        data_type = torch.cat([ torch.LongTensor(sample['data_types']) for sample in batch],dim=0)

        return {
            "input_ids":user_id,
            "labels":user_id,
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "answers": answers,
            "neg_samples": neg_samples,
            "data_type": data_type,
        }

class LLMTextDataset(LLMDataset):
    # no augmentation
    def __init__(self, item_size,llama_tokenizer, max_seq_length, max_title_length, data_type='train', csv_path=''):
        # self.args = args
        self.df = pd.read_csv(csv_path)
        self.item_size = item_size
        # self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = max_seq_length
        self.df['asin'] = self.df['asin'].apply(json.loads)
        self.max_len2 = max_title_length
        # 现在我们有了包含真实列表的DataFrame列，我们可以找出所有数字的范围
        self.m_item = max(self.df['asin'].explode())+1

        self.df['title'] = self.df['title'].apply(parse_list)
        self.reviewerID = self.df['reviewerID'].tolist()#self.df['reviewerID'].tolist()
        self.llama_tokenizer = llama_tokenizer
        self.user_seq = self.df['asin'].tolist()#self.df['asin_x'].tolist()
        self.title = self.df['title'].tolist()
        self.tail_len = self.__return_tailed_length__(self.df['asin'])
        self.item_pool = self.__build_i_set__(self.user_seq)     
        self.seq_len = max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        titles = self.title[index]
        assert len(items) == len(titles)
        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [items[-3]]
            titles = titles[:-3]

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
            titles = titles[:-2]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
            titles = titles[:-1]

        # print(items,input_ids,target_pos,answer)
        
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.item_size))

        # print(len(target_neg))
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        # pad_len2 = self.max_len2 - len(titles)
        # titles = ["<pad>"] * pad_len2 + titles
        # try:
        titles_str = ' '.join(titles)
        # except:
        #     print(titles)
        title_ids, title_mask = self.llama_tokenizer(titles_str,truncation=True,max_length=self.max_len2,padding='max_length',return_tensors='pt', add_special_tokens=False).values()
        neg_items_set = self.item_pool - seq_set
        neg_samples = random.sample(neg_items_set, self.item_size)

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        sample = dict()
        if self.data_type == "train":
            sample['user_id'] = np.array([user_id])
            sample['input_ids'] = np.array([input_ids])
            sample['target_pos'] = np.array([target_pos])
            sample['target_neg'] = np.array([target_neg])
            sample['answer'] = np.array([answer])
            sample['neg_samples'] = np.array([neg_samples])
            sample['titles'] = title_ids
            sample['titles_mask'] = title_mask
            sample['data_types'] = np.array([0]) # train:0,val/test:1
            return sample
        else:
            sample['user_id'] = np.array([user_id])
            sample['input_ids'] = np.array([input_ids])
            sample['target_pos'] = np.array([target_pos])
            sample['target_neg'] = np.array([target_neg])
            sample['answer'] = np.array([answer])
            sample['neg_samples'] = np.array([neg_samples])
            sample['titles'] = title_ids
            sample['titles_mask'] = title_mask
            sample['data_types'] = np.array([1]) # train:0,val/test:1
            return sample

    def __len__(self):
        return len(self.user_seq)

@dataclass
class SequentialCollator2:
    def __call__(self, batch) -> dict:
        user_id = torch.cat([ torch.LongTensor(sample['user_id']) for sample in batch],dim=0)
        inputs = torch.cat([ torch.LongTensor(sample['input_ids']) for sample in batch],dim=0)
        answers = torch.cat([ torch.LongTensor(sample['answer']) for sample in batch],dim=0)
        neg_samples = torch.cat([ torch.LongTensor(sample['neg_samples']) for sample in batch],dim=0)
        titles = torch.cat([ torch.LongTensor(sample['titles']) for sample in batch],dim=0)
        titles_mask = torch.cat([ torch.LongTensor(sample['titles_mask']) for sample in batch],dim=0)
        inputs_mask = (inputs > 0)
        data_type = torch.cat([ torch.LongTensor(sample['data_types']) for sample in batch],dim=0)

        return {
            "input_ids":user_id,
            "labels":user_id,
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "answers": answers,
            "neg_samples": neg_samples,
            "titles":titles,
            "titles_mask":titles_mask,
            "data_type": data_type,
        }