import copy
import sys
import os
import torch
import numpy as np

def add_noise(noise, labels1, labels2, num_l):
    if noise == 'uniform-positive':
        for i in range(len(labels1)):
            for j in labels1[i]:
                if j in {0,1,2,3}: continue
                rnd = np.random.uniform()
                if rnd < noise_level:
                    if len(labels2[i]) > 3:
                        labels2[i].remove(j)
                    else:
                        print("WARNING: TOO FEW LABELS TO REMOVE")
    elif noise == 'one-positive':
        for i in range(len(labels1)):
            for j in labels1[i]:
                if j in {0,1,2,3}: continue
                labels2[i] = [2,3]
                labels2[i].insert(1, np.random.choice(labels1[i][1:-1])) 
    elif noise == 'uniform':
        for i in range(len(labels1)):
            for j in range(num_l):
                if j in {0,1,2,3}: continue
                rnd = np.random.uniform()
                if rnd < noise_level:
                    if j in labels1[i]:
                        if len(labels2[i]) > 3:
                            labels2[i].remove(j)
                        else:
                            print("WARNING: TOO FEW LABELS TO REMOVE")
                    else:
                        labels2[i].insert(-2, j)
    elif noise == 'combined':
        for i in range(len(labels1)):
            rnd1 = np.random.uniform()
            if rnd1 < 1/3:
                for j in labels1[i]:
                    if j in {0,1,2,3}: continue
                    rnd = np.random.uniform() + 0.5
                    if rnd < noise_level:
                        if len(labels2[i]) > 3:
                            labels2[i].remove(j)
                        else:
                            print("WARNING: TOO FEW LABELS TO REMOVE")
            elif 1/3 <= rnd1 < 2/3:
                for j in labels1[i]:
                    if j in {0,1,2,3}: continue
                    labels2[i] = [2,3]
                    labels2[i].insert(1, np.random.choice(labels1[i][1:-1])) 
            else:
                for j in range(num_l):
                    if j in {0,1,2,3}: continue
                    rnd = np.random.uniform() + 0.9
                    if rnd < noise_level:
                        if j in labels1[i]:
                            if len(labels2[i]) > 3:
                                labels2[i].remove(j)
                            else:
                                print("WARNING: TOO FEW LABELS TO REMOVE")
                        else:
                            labels2[i].insert(-2, j)

dataset = sys.argv[1]
noise_level = float(sys.argv[2])
noise_type = sys.argv[3]

original = torch.load('%s/train_valid_test.pt' % dataset)
modified = copy.deepcopy(original)
add_noise(noise_type, original['train']['tgt'], modified['train']['tgt'], len(original['dict']['tgt']))
add_noise(noise_type, original['valid']['tgt'], modified['valid']['tgt'], len(original['dict']['tgt']))


if not os.path.isdir('%s_%.2f_%s' % (dataset, noise_level, noise_type)):
    os.mkdir('%s_%.2f_%s' % (dataset, noise_level, noise_type))
torch.save(modified, '%s_%.2f_%s/train_valid_test.pt' % (dataset, noise_level, noise_type))

tot1 = 0
tot2 = 0
for i in range(len(original['train']['tgt'])):
    tot1 += len(original['train']['tgt'][i]) - 2
    tot2 += len(modified['train']['tgt'][i]) - 2
    print(i)
    print(original['train']['tgt'][i])
    print(modified['train']['tgt'][i])

print("original positive:", tot1/len(original['train']['tgt']))
print("noise positive:", tot2/len(original['train']['tgt']))
