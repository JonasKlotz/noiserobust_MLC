import numpy as np
import torch
import sys
import copy

def get_embedding(d, glove):
    #res = [[]] * len(d)
    res = list()
    for _ in range(len(d)):
        res.append([])
    idx2word = {}
    for word in d.keys():
        idx2word[d[word]] = word
    d2 = flatten(d)

    with open(glove, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            if tmp[0] in d2.keys():
                res[d2[tmp[0]]].append([float(x) for x in tmp[1:]])
    # need to process this first in order to get the correct mean and std
    for i, emb in enumerate(res):
        if len(emb) > 0:
            emb = np.array(emb)
            res[i] = np.mean(emb, axis=0)
        else:
            res[i] = np.zeros((300,))
    flattened = [item for sublist in res for item in sublist]
    mean, std = np.mean(flattened), np.std(flattened)
    for i, emb in enumerate(res):
        if len(emb) == 0:
            print(idx2word[i], "does not exist in the pretrained embedding.")
            res[i] = np.random.normal(loc=mean, scale=std, size=(300,))
    return np.array(res)

def flatten(d):
    new = dict()
    for key in d.keys():
        if '_' in key:
            tmp = key.split('_')
            tmp = [k.lower() for k in tmp]
            for k in tmp:
                new[k] = d[key]
        else:
            new[key] = d[key]
    return new
                

dataset = sys.argv[1]
data = torch.load('data/%s/train_valid_test.pt' % dataset)
glove_fn = 'glove/glove.840B.300d.txt'

src_dict = data['dict']['src']
tgt_dict = data['dict']['tgt']

#a = get_embedding(src_dict, glove_fn)
#print(a)
#print(a.shape)
#np.save('data/%s/src_word_embedding.npy' % dataset, a)
b = get_embedding(tgt_dict, glove_fn)
#print(b)
print(b.shape)
np.save('data/%s/tgt_word_embedding.npy' % dataset, b)
