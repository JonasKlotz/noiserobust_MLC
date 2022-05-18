import numpy as np
import torch
import sys

def get_embedding(d, glove):
    res = [[0 for _ in range(300)]] * len(d)
    idx2word = {}
    for word in d.keys():
        idx2word[d[word]] = word

    with open(glove, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            if tmp[0] in d.keys():
                res[d[tmp[0]]] = [float(x) for x in tmp[1:]]
    mean, std = np.mean(res), np.std(res)
    for i, emb in enumerate(res):
        if np.sum(emb) == 0:
            print(idx2word[i], "does not exist in the pretrained embedding.")
            res[i] = np.random.normal(loc=mean, scale=std, size=(300,))
    return np.array(res)
                

dataset = sys.argv[1]
data = torch.load('data/%s/train_valid_test.pt' % dataset)
glove_fn = 'glove/glove.840B.300d.txt'

src_dict = data['dict']['src']
tgt_dict = data['dict']['tgt']

a = get_embedding(src_dict, glove_fn)
print(a)
print(a.shape)
np.save('data/%s/src_word_embedding.npy' % dataset, a)
b = get_embedding(tgt_dict, glove_fn)
print(b)
print(b.shape)
np.save('data/%s/tgt_word_embedding.npy' % dataset, b)
