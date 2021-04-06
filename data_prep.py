"""# Loading Training Data"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import numpy as np
from torchnlp.datasets import iwslt_dataset

train_text = iwslt_dataset(directory="data/", train=True, check_files=[], url="http://cs.cornell.edu/~junxiong/data.tgz")
validation_text = iwslt_dataset(directory="data/", dev=True, check_files=[], url="http://cs.cornell.edu/~junxiong/data.tgz")
test_text = iwslt_dataset(directory="data/", test=True, check_files=[], url="http://cs.cornell.edu/~junxiong/data.tgz")

print(train_text[:3])

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torchtext

de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')

def build_vocab(text, tokenizer):
  counter = Counter()
  for string_ in text:
    counter.update(tokenizer(string_))
  return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq=2)

de_vocab = build_vocab(list(map(lambda x: x['de'], train_text)), de_tokenizer)
en_vocab = build_vocab(list(map(lambda x: x['en'], train_text)), en_tokenizer)

print(len(en_vocab))
print(len(de_vocab))

def data_process(text):
  data = []
  for sen in text:
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(sen['en'])],
                            dtype=torch.long)
    de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(sen['de'])],
                            dtype=torch.long)
    data.append((en_tensor_, de_tensor_))
  return data

train_data = data_process(train_text)
val_data = data_process(validation_text)
test_data = data_process(test_text)

# print(test_data[3])
# print(val_data[3])
# print(train_data[3])
# print(len(de_vocab))
# print(len(en_vocab))
for (i, j) in train_data:
  if torch.max(i) >= 36622:
    print(i)

# batchify data
# BATCH_SIZE = 25000
BATCH_SIZE = 10

EN_PAD_IDX = en_vocab['<pad>']
EN_BOS_IDX = en_vocab['<bos>']
EN_EOS_IDX = en_vocab['<eos>']
DE_PAD_IDX = de_vocab['<pad>']
DE_BOS_IDX = de_vocab['<bos>']
DE_EOS_IDX = de_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([DE_BOS_IDX]), de_item, torch.tensor([DE_EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([EN_BOS_IDX]), en_item, torch.tensor([EN_EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=DE_PAD_IDX).transpose(0, 1)
  en_batch = pad_sequence(en_batch, padding_value=EN_PAD_IDX).transpose(0, 1)
  return de_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)

src_max_sen_len = max(max(i.shape[0], j.shape[0]) for (i, j) in train_data)
print(src_max_sen_len)
print(max(i.shape[0] for (i, j) in train_data))
print(max(j.shape[0] for (i, j) in train_data))

print(en_vocab['<unk>'])
print(en_vocab['<pad>'])
print(en_vocab['<bos>'])
print(en_vocab['<eos>'])

print(de_vocab['<unk>'])
print(de_vocab['<pad>'])
print(de_vocab['<bos>'])
print(de_vocab['<eos>'])