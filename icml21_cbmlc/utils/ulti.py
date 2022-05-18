import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import os, sys
import timeit 

from scipy import stats


def Species_Acc(pred, Y): #the smaller the better
    return np.mean(np.abs(pred - Y))

def Species_Dis(pred, Y): #the larger the better
    res = []
    for i in range(pred.shape[1]):
        try:
            auc = roc_auc_score(Y[:, i] ,pred[:, i])
            res.append(auc)
        except:
            res.append(1.0) #print("AUC nan", i, np.mean(Y[:, i]), np.mean(pred[:, i]))
        
    return np.mean(res)

def Species_Cali(pred, Y):
    res = []
    for j in range(pred.shape[1]):
        p = pred[:, j]
        y = Y[:, j]

        bin1 = np.zeros(10)
        bin2 = np.zeros(10)
        th = np.zeros(10)

        for k in range(10):
            th[k] = np.percentile(p, (k+1)*10)

        for i in range(p.shape[0]):
            for k in range(10):
                if (p[i] <= th[k]):
                    bin1[k] += p[i]
                    bin2[k] += y[i]
                    break

        diff = np.sum(np.abs(bin1 - bin2))
        #print(bin1)
        #print(bin2)
        res.append(diff)
    return np.mean(res)

def Species_Prec(pred, Y): #the smaller the better
    return np.mean(np.sqrt(pred * (1 - pred)))

def Richness_Acc(pred, Y): #the smaller the better
    return np.sqrt(np.mean((np.sum(pred, axis = 1)-np.sum(Y, axis = 1)) ** 2))

def Richness_Dis(pred, Y): #the larger the better
    return stats.spearmanr(np.sum(pred, axis = 1), np.sum(Y, axis = 1))[0]

def Richness_Cali(pred, Y): #the smaller the better
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))
    richness = np.sum(samples, axis = 2) #100, n
    gt_richness = np.sum(Y, axis = 1)

    res = []

    for i in range(pred.shape[0]):
        if (gt_richness[i] <= np.percentile(richness[:, i], 75) and gt_richness[i] >= np.percentile(richness[:, i], 25)):
            res.append(1)
        else:
            res.append(0)
    p = np.mean(res)
    return np.abs(p - 0.5)

def Richness_Prec(pred, Y): #the smaller the better
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))

    return np.mean(np.std(np.sum(samples, axis = 2), axis = 0))

def Beta_SOR(x, y):
    if (np.sum(x * y) == 0 and np.sum(x + y) == 0):
        return 0

    return 1 - 2 * np.sum(x * y)/np.maximum(np.sum(x + y), 1e-9)

def Beta_SIM(x, y):
    if (np.sum(x * y) == 0 and np.minimum(np.sum(x), np.sum(y)) == 0):
        return 0
    return 1 - np.sum(x * y)/np.maximum(np.minimum(np.sum(x), np.sum(y)), 1e-9)

def Beta_NES(x, y):
    return Beta_SOR(x, y) - Beta_SIM(x, y)

def get_dissim(pred, Y):
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))

    pairs = []
    for i in range(300):
        x = np.random.randint(pred.shape[0])
        y = np.random.randint(pred.shape[0])
        pairs.append([x, y])


    SOR = np.zeros((300, 100))
    SIM = np.zeros((300, 100))
    NES = np.zeros((300, 100))

    gt_SOR = []
    gt_SIM = []
    gt_NES = []
    for i in range(300):
        x, y = pairs[i]
        for j in range(100):
            SOR[i][j] = Beta_SOR(samples[j][x], samples[j][y])
            SIM[i][j] = Beta_SIM(samples[j][x], samples[j][y])
            NES[i][j] = Beta_NES(samples[j][x], samples[j][y])

        gt_SOR.append(Beta_SOR(Y[x], Y[y]))
        gt_SIM.append(Beta_SIM(Y[x], Y[y]))
        gt_NES.append(Beta_NES(Y[x], Y[y]))
    return SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES

def Community_Acc(pred, Y): #the smaller the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return np.sqrt(np.mean((np.mean(SOR, axis = 1) - gt_SOR)**2)),\
    np.sqrt(np.mean((np.mean(SIM, axis = 1) - gt_SIM)**2)),\
    np.sqrt(np.mean((np.mean(NES, axis = 1) - gt_NES)**2))

def Community_Dis(pred, Y): #the larger the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return stats.spearmanr(np.mean(SOR, axis = 1), gt_SOR)[0],\
    stats.spearmanr(np.mean(SIM, axis = 1), gt_SIM)[0],\
    stats.spearmanr(np.mean(NES, axis = 1), gt_NES)[0]

def Community_Cali(pred, Y):
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    tmp1 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(SOR, 25, axis = 1), gt_SOR),\
     np.greater_equal(np.percentile(SOR, 75, axis = 1),gt_SOR)).astype("float")) - 0.5)

    tmp2 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(SIM, 25, axis = 1), gt_SIM),\
     np.greater_equal(np.percentile(SIM, 75, axis = 1),gt_SIM)).astype("float")) - 0.5)

    tmp3 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(NES, 25, axis = 1), gt_NES),\
     np.greater_equal(np.percentile(NES, 75, axis = 1),gt_NES)).astype("float")) - 0.5)

    return tmp1, tmp2, tmp3

def Community_Prec(pred, Y): #the smaller the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return np.mean(np.std(SOR, axis = 1)), \
    np.mean(np.std(SIM, axis = 1)), \
    np.mean(np.std(NES, axis = 1))


metrics = [Species_Acc, Species_Dis, Species_Cali, Species_Prec, \
        Richness_Acc, Richness_Dis, Richness_Cali, Richness_Prec, 
        Community_Acc, Community_Dis, Community_Cali, Community_Prec]

metric_names = ["Species_Acc", "Species_Dis", "Species_Cali", "Species_Prec", \
                "Richness_Acc", "Richness_Dis", "Richness_Cali", "Richness_Prec", \
                "Community_Acc", "Community_Dis", "Community_Cali", "Community_Prec"]
#xxx_Dis the larger the better
#others the smaller the better
#community_xxxx return 3 scores 

def compute_all(pred, Y):
    res = dict()
    for i in range(len(metrics)):
        res[metric_names[i]] = metrics[i](pred, Y)
    return res
