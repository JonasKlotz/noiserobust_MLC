import voc
import nuswide
import coco
import torchvision.transforms as transforms
import numpy as np
import torch
import sys

def collect(d):
    srcs, tgts = [], []
    #for i in range(len(dataset)):
    for i in range(10):
        (source,_), _, (target,_) = d[i]
        tgt = target.cpu().data.numpy()
        tgt = [i+4 for i in range(len(tgt)) if tgt[i] > 0]
        tgts.append([2]+tgt+[3])
        source = np.array(source)
        srcs.append(source)
    return srcs, tgts

dataset_name = sys.argv[1]
if dataset_name == 'voc':
    dataset = voc.Voc2007Classification('/mnt/beegfs/bulk/mirror/wz346/ML-GCN/data/voc/', 'trainval')
    tdataset = voc.Voc2007Classification('/mnt/beegfs/bulk/mirror/wz346/ML-GCN/data/voc/', 'test')
elif dataset_name == 'nuswide':
    dataset = nuswide.NUSWIDEClassification('/mnt/beegfs/bulk/mirror/wz346/noisy_mlc/data/nuswide/', 'trainval')
    tdataset = nuswide.NUSWIDEClassification('/mnt/beegfs/bulk/mirror/wz346/noisy_mlc/data/nuswide/', 'test')
elif dataset_name == 'coco':
    dataset = coco.COCO2014('/mnt/beegfs/bulk/mirror/wz346/ML-GCN/data/coco/', 'trainval')
    tdataset = coco.COCO2014('/mnt/beegfs/bulk/mirror/wz346/ML-GCN/data/coco/', 'test')

res = dict()
res['train'], res['valid'], res['test'] = dict(), dict(), dict()
res['dict'] = dict()
src_dict, tgt_dict = None, voc.object_categories
tgt_dict = {tgt_dict[i]: i+4 for i in range(len(tgt_dict))}

res['train']['src'], res['train']['tgt'] = collect(dataset)
res['valid']['src'], res['valid']['tgt'] = collect(tdataset)
res['test']['src'], res['test']['tgt'] = [[]], [[]]
aux = {'<s>': 2, '</s>': 3, '<blank>': 0, '<unk>': 1}
res['dict']['tgt'] = {**tgt_dict, **aux}
res['dict']['src'] = src_dict

torch.save(res, './data/%s/train_valid_test.pt' % (dataset_name))
