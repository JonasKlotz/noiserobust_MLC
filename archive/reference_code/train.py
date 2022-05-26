import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import utils.evals as evals
import utils.utils as utils
from utils.data_loader import process_data
import torch, torch.nn as nn, torch.nn.functional as F
from torch import linalg as LA
import lamp.Constants as Constants
from lamp.Models import LAMP
from lamp.Translator import translate
from config_args import config_args,get_args
from pdb import set_trace as stop
from tqdm import tqdm
from torch.distributions import Categorical



def build_multi_classification_loss(predictions, labels):
    shape = tuple(labels.shape)
    labels = labels.float()
    y_i = torch.eq(labels, torch.ones(shape).cuda())
    y_not_i = torch.eq(labels, torch.zeros(shape).cuda())

    truth_matrix = pairwise_and(y_i, y_not_i).float()
    sub_matrix = pairwise_sub(predictions, predictions)
    exp_matrix = torch.exp(-5*sub_matrix)
    sparse_matrix = exp_matrix * truth_matrix
    sums = torch.sum(sparse_matrix, dim=[1,2])
    y_i_sizes = torch.sum(y_i.float())
    y_i_bar_sizes = torch.sum(y_not_i.float())
    normalizers = y_i_sizes * y_i_bar_sizes
    loss = torch.div(sums, 5*normalizers) # 100*128  divide  128
    zero = torch.zeros_like(loss) # 100*128 zeros
    loss = torch.where(torch.logical_or(torch.isinf(loss), torch.isnan(loss)), zero, loss)
    loss = torch.mean(loss)
    return loss

def pairwise_and(a, b):
    column = torch.unsqueeze(a, 2)
    row = torch.unsqueeze(b, 1)
    return torch.logical_and(column, row)

def pairwise_sub(a, b):
    column = torch.unsqueeze(a, 2)
    row = torch.unsqueeze(b, 1)
    return column - row

def sparsity_constraints(label_matrix, posratio, negratio):
    assert label_matrix.shape[0] == label_matrix.shape[1]
    negpairs = []
    pospairs = []
    for i in range(len(label_matrix)):
        sortedarr= np.sort(label_matrix[i])
        if posratio == 0:
            posthres = 1e10
        else:
            posthres = sortedarr[-int(posratio*len(label_matrix))]
        if negratio == 0:
            negthres = -1
        else:
            negthres = sortedarr[int(negratio*len(label_matrix))]
        for j in range(len(label_matrix)):
            if i < j:
                if label_matrix[i][j] >= posthres:
                    pospairs.append((i,j))
                elif label_matrix[i][j] <= negthres:
                    negpairs.append((i,j))
    return pospairs, negpairs

def compute_spar(pred, pospairs, negpairs, prob):
    if len(pospairs) + len(negpairs) == 0:
        return torch.tensor(0.), 0
    tot = 0.
    cnt = 0
    for (i, j) in pospairs:
        tmp_prob = np.random.rand()
        if tmp_prob < prob:
            continue
        cnt += 1
        probs = torch.stack((pred[..., i], pred[..., j])).T
        tot += -Categorical(probs=probs).entropy().mean()
    for (i, j) in negpairs:
        tmp_prob = np.random.rand()
        if tmp_prob < prob:
            continue
        cnt += 1
        probs = torch.stack((pred[..., i], pred[..., j])).T
        tot += Categorical(probs=probs).entropy().mean()
    if cnt == 0:
        return torch.tensor(0.), 0
    else:
        tot /= cnt
        return tot, cnt

def train_epoch(model,train_data, crit, optimizer,epoch,data_dict,opt,label_adj_matrix):
    model.train()
    if opt.spar_loss:
        pospairs, negpairs = sparsity_constraints(label_adj_matrix[1], posratio=opt.spar_posratio, negratio=opt.spar_negratio)

    out_len = (opt.tgt_vocab_size) if opt.binary_relevance else (opt.tgt_vocab_size-1)

    if 'resnet' in opt.encoder:
        all_predictions = torch.zeros(len(train_data.dataset),out_len)
        all_targets = torch.zeros(len(train_data.dataset),out_len)
    else:
        all_predictions = torch.zeros(len(train_data._src_insts),out_len)
        all_targets = torch.zeros(len(train_data._src_insts),out_len)

    
    batch_idx,batch_size = 0,opt.batch_size
    loss_total,d_total,d_fake_total,g_total = 0,0,0,0
    entropy_total = 0
    ranking_total = 0
    spar_total, spar_cnt = 0, 0
    total_tgt_emb_l2 = 0
    
    for batch in tqdm(train_data, mininterval=0.5,desc='(Training)', leave=False):
        src,adj,tgt = batch
        loss,d_loss = 0,0
        gold = tgt[0][:, 1:]

        if opt.binary_relevance:
            if 'resnet' not in opt.encoder:
                gold_binary = utils.get_gold_binary(gold.data.cpu(), opt.tgt_vocab_size).cuda()
            else:
                gold_binary = tgt[0].cuda()
                src[0] = src[0].cuda()
            optimizer.zero_grad()

            pred,enc_output,*results = model(src,adj,None,gold_binary,return_attns=opt.attns_loss,int_preds=opt.int_preds)
            norm_pred = F.sigmoid(pred)
            tmp = crit(pred, gold_binary) 
            loss += tmp
            loss_total += tmp.item()
            if opt.ranking_loss:
                ranking_loss = opt.ranking_weight * build_multi_classification_loss(norm_pred, gold_binary)
                loss += ranking_loss
                ranking_total += franking_loss.item()
            if opt.entropy_loss: 
                pred_probs = F.softmax(pred, dim=1)
                entropy = opt.ent_coeff * Categorical(probs=pred_probs).entropy().mean()
                loss += entropy
                entropy_total += entropy.item()
            if opt.spar_loss:
                spar_loss, cnt = compute_spar(norm_pred, pospairs, negpairs, opt.spar_prob)
                spar_loss *= opt.spar_coeff
                loss += spar_loss 
                spar_cnt += cnt
                spar_total += spar_loss.item()
            if opt.tgt_emb_l2 > 0:
                if opt.multi_gpu:
                    tgt_emb_diff = opt.tgt_emb_l2 * LA.norm(opt.ori_tgt_emb - model.module.decoder.tgt_word_emb.weight)
                else:
                    tgt_emb_diff = opt.tgt_emb_l2 * LA.norm(opt.ori_tgt_emb - model.decoder.tgt_word_emb.weight)
                loss += tgt_emb_diff
                total_tgt_emb_l2 += tgt_emb_diff.item()

            if opt.int_preds and not opt.matching_mlp:
                for i in range(len(results[0])):
                    tmp2 = crit(results[0][i], gold_binary)
                    loss_total += tmp2.item()
                    loss += (opt.int_pred_weight)*tmp2
            if epoch == opt.thresh1:
                opt.init_model = copy.deepcopy(model)
            loss.backward()
            optimizer.step()
            tgt_out = gold_binary.data
            try:
                pred_out = norm_pred.data
            except:
                pred_out = norm_fpred.data

        else: 
            # Non Binary Outputs
            optimizer.zero_grad()
            pred,enc_output,*results = model(src,adj,tgt,None,int_preds=opt.int_preds)
            loss = crit(F.log_softmax(pred), gold.contiguous().view(-1))
            pred = F.softmax(pred,dim=1)
            pred_vals,pred_idxs = pred.max(1)
            pred_vals = pred_vals.view(gold.size()).data.cpu()
            pred_idxs = pred_idxs.view(gold.size()).data.cpu()
            pred_out = torch.zeros(pred_vals.size(0),pred.size(1)).scatter_(1,pred_idxs.long(),pred_vals)
            tgt_out = torch.zeros(pred_vals.size(0),pred.size(1)).scatter_(1,gold.data.cpu().long(),torch.ones(pred_vals.size()))
            pred_out = pred_out[:,1:]
            tgt_out = tgt_out[:,1:]
            loss.backward()
            optimizer.step()

        
        ## Updates ##
        start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
        all_predictions[start_idx:end_idx] = pred_out
        all_targets[start_idx:end_idx] = tgt_out
        batch_idx +=1

    freq = len(all_targets) / batch_size
    print("main loss: %.5f" % (loss_total/freq))
    print("tgtemb l2: %.5f" % (total_tgt_emb_l2/freq))
    print("entropy: %.5f" % (entropy_total/freq))
    print("ranking: %.5f" % (ranking_total/freq))
    print("sparsity: %.5f, #constraints: %d" % (spar_total/freq, spar_cnt/freq))
        
    
    return all_predictions, all_targets, loss_total
