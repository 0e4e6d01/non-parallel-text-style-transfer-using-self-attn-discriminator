import re
import linecache
import torch
from torch.utils.data import Dataset
from math import log


# Dataset for classifier
class ClassifierDataset(Dataset):
    
    def __init__(self, src, lab):
        assert len(src) == len(lab)
        self.src = src
        self.lab = lab
        self.size = len(self.src)
        # print('self.size is', self.size)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, index):
        # print(index)
        return self.src[index], self.lab[index]

# Dataset for classifier
class ClassifierDatasetWithoutLab(Dataset):
    
    def __init__(self, src):
        self.src = src
        self.size = len(self.src)
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # print(index)
        return self.src[index]

# Dataset for classifier
class PairDataset(Dataset):
    
    def __init__(self, src, ano, lab):
        """
        ano for another
        """
        assert len(src) == len(lab)
        assert len(src) == len(ano)
        self.src = src
        self.ano = ano
        self.lab = lab
        self.size = len(self.src)
        # print('self.size is', self.size)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, index):
        # print(index)
        return self.src[index], self.ano[index], self.lab[index]


# Collating function for classifier
def ClassifierCollate(data):
    src, lab = zip(*data)
    return torch.LongTensor(src), torch.LongTensor(lab)

def ClassifierPaddingCollate(data):
    # print('data is', data)
    src, lab = zip(*data)
    # print('src is', src)
    # print('lab is', lab)
    
    src_len = [len(s) for s in src]
    
    # shape of src_pad will be [batch_size, max_src_len]
    src_pad = torch.zeros(len(src), max(src_len), dtype=torch.int64)
    
    for i, s in enumerate(src):
        src_pad[i, :src_len[i]] = torch.LongTensor(s)
    # print('src_pad is', src_pad)
    # print('lab is', lab)
    # print('src_len is', src_len)
    
    return src_pad, torch.LongTensor(lab), torch.IntTensor(src_len)

# Collating function for classifier
def ClassifierPaddingCollateWithoutLab(data):
    # print('data is', data)
    src = data
    # print('src is', src)
    
    src_len = [len(s) for s in src]
    
    # shape of src_pad will be [batch_size, max_src_len]
    src_pad = torch.zeros(len(src), max(src_len), dtype=torch.int64)
    
    for i, s in enumerate(src):
        src_pad[i, :src_len[i]] = torch.LongTensor(s)
    # print('src_pad is', src_pad)
    # print('lab is', lab)
    # print('src_len is', src_len)
    
    return src_pad, torch.IntTensor(src_len)


def PairPaddingCollate(data):
    # print('data is', data)
    src, ano, lab = zip(*data)
    # print('src is', src)
    # print('lab is', lab)
    
    src_len = [len(s) for s in src]
    ano_len = [len(a) for a in ano]
    
    # shape of src_pad will be [batch_size, max_src_len]
    src_pad = torch.zeros(len(src), max(src_len), dtype=torch.int64)
    ano_pad = torch.zeros(len(ano), max(ano_len), dtype=torch.int64)
    
    for i, sample in enumerate(zip(src, ano)):
        s, a = sample
        src_pad[i, :src_len[i]] = torch.LongTensor(s)
        ano_pad[i, :ano_len[i]] = torch.LongTensor(a)
    # print('src_pad is', src_pad)
    # print('lab is', lab)
    # print('src_len is', src_len)
    
    return src_pad, ano_pad, torch.LongTensor(lab), torch.IntTensor(src_len),\
        torch.IntTensor(ano_len)

def RespectivelyPaddingCollate(data):
    # print('data is', data)
    src, lab = zip(*data)
    # print('src is', src)
    # print('lab is', lab)
    
    src_0 = []
    src_1 = []
    src_0_len = []
    src_1_len = []
    for s, l in zip(src, lab):
        if l:
            src_1.append(s)
            src_1_len.append(len(s))
        else:
            src_0.append(s)
            src_0_len.append(len(s))
    lab_0 = [0] * len(src_0)
    lab_1 = [1] * len(src_1)
    
    # shape of src_pad will be [batch_size, max_src_len]
    if src_0_len:
        src_0_pad = torch.zeros(len(src_0), max(src_0_len), dtype=torch.int64)
    else:
        src_0_pad = torch.LongTensor([])
    if src_1_len:
        src_1_pad = torch.zeros(len(src_1), max(src_1_len), dtype=torch.int64)
    else:
        src_1_pad = torch.LongTensor([])
    
    for i, s in enumerate(src_0):
        src_0_pad[i, :src_0_len[i]] = torch.LongTensor(s)
    for i, s in enumerate(src_1):
        src_1_pad[i, :src_1_len[i]] = torch.LongTensor(s)
    # print('src_pad is', src_pad)
    # print('lab is', lab)
    # print('src_len is', src_len)
    
    return src_0_pad, src_1_pad, torch.LongTensor(lab_0), torch.LongTensor(lab_1),\
            torch.IntTensor(src_0_len), torch.IntTensor(src_1_len)


def MaskingPaddingCollate(data):
    # print('data is', data)
    src, lab = zip(*data)
    # print('src is', src)
    # print('lab is', lab)
    
    src_len = [len(s) for s in src]
    
    # shape of src_pad will be [batch_size, max_src_len]
    src_pad = torch.zeros(len(src), max(src_len), dtype=torch.int64)
    
    for i, s in enumerate(src):
        src_pad[i, :src_len[i]] = torch.LongTensor(s)
    # print('src_pad is', src_pad)
    # print('lab is', lab)
    # print('src_len is', src_len)
    src_pad_mask = src_pad==0
    
    return src_pad, torch.LongTensor(lab), src_pad_mask

def NoLabMaskingPaddingCollate(data):
    # print('data is', data)
    src = data
    # print('src is', src)
    
    src_len = [len(s) for s in src]
    
    # shape of src_pad will be [batch_size, max_src_len]
    src_pad = torch.zeros(len(src), max(src_len), dtype=torch.int64)
    
    for i, s in enumerate(src):
        src_pad[i, :src_len[i]] = torch.LongTensor(s)
    # print('src_pad is', src_pad)
    # print('lab is', lab)
    # print('src_len is', src_len)
    src_pad_mask = src_pad==0
    
    return src_pad, src_pad_mask