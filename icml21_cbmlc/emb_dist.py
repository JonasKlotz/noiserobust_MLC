import numpy as np
from numpy import linalg as LA
import torch
import sys


params = torch.load(sys.argv[2])
embs = params['model']['module.tgt_word_proj.weight'].cpu().data.numpy()
#embs = params['model']['tgt_word_proj.weight'].cpu().data.numpy()
m1 = LA.norm(np.amax(embs, axis=0))
#m1 = LA.norm(np.mean(embs, axis=0))

params2 = torch.load(sys.argv[3])
bembs = params2['model']['module.tgt_word_proj.weight'].cpu().data.numpy()
#bembs = params2['model']['tgt_word_proj.weight'].cpu().data.numpy()
m2 = LA.norm(np.amax(bembs, axis=0))
#m2 = LA.norm(np.mean(bembs, axis=0))

adjs = np.load("adjs/%s_adjs.npy" % sys.argv[1])

(xs, ys) = np.unravel_index(np.argsort(adjs, axis=None), adjs.shape)
xs = xs[::-1][:100]
ys = ys[::-1][:100]

tot1, tot2 = 0, 0
for i in range(100):
    dist1 = LA.norm(embs[xs[i]] - embs[ys[i]])/m1
    tot1 += dist1
    dist2 = LA.norm(bembs[xs[i]] - bembs[ys[i]])/m2
    tot2 += dist2

print(tot1/100, tot2/100)
