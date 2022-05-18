import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import torch
from lamp.Models import LAMP
from test import test_epoch
import utils.evals as evals
from config_args import config_args,get_args
from utils.data_loader import process_data

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)
data = torch.load(opt.data)
train_data,valid_data,test_data,label_adj_matrix,opt = process_data(data,opt)
dataset = opt.dataset.split('/')[-1]
if opt.load_src_embedding:
    src_emb = torch.from_numpy(np.load('data/%s/src_word_embedding.npy' % dataset))
    src_emb = src_emb.float()
else:
    src_emb = None
if opt.load_tgt_embedding:
    tgt_emb = torch.from_numpy(np.load('data/%s/tgt_word_embedding.npy' % dataset))
    tgt_emb = tgt_emb[4:]
    tgt_emb = tgt_emb.float()
    if opt.tgt_emb_l2 > 0:
        opt.ori_tgt_emb = copy.deepcopy(tgt_emb).cuda()
else:
    tgt_emb = None
model = LAMP(
    opt.src_vocab_size,
    opt.tgt_vocab_size,
    opt.max_token_seq_len_e,
    opt.max_token_seq_len_d,
    proj_share_weight=opt.proj_share_weight,
    embs_share_weight=opt.embs_share_weight,
    d_k=opt.d_k,
    d_v=opt.d_v,
    d_model=opt.d_model,
    d_word_vec=opt.d_word_vec,
    d_inner_hid=opt.d_inner_hid,
    n_layers_enc=opt.n_layers_enc,
    n_layers_dec=opt.n_layers_dec,
    n_head=opt.n_head,
    n_head2=opt.n_head2,
    dropout=opt.dropout,
    dec_dropout=opt.dec_dropout,
    dec_dropout2=opt.dec_dropout2,
    encoder=opt.encoder,
    decoder=opt.decoder,
    enc_transform=opt.enc_transform,
    onehot=opt.onehot,
    no_enc_pos_embedding=opt.no_enc_pos_embedding,
    no_dec_self_att=opt.no_dec_self_att,
    loss=opt.loss,
    label_adj_matrix=label_adj_matrix[0],
    attn_type=opt.attn_type,
    label_mask=opt.label_mask,
    matching_mlp=opt.matching_mlp,
    graph_conv=opt.graph_conv,
    int_preds=opt.int_preds,
    src_emb=src_emb,
    tgt_emb=tgt_emb,
    freeze_emb=opt.freeze_emb)
model = model.cuda()
if opt.load_pretrained:     
    checkpoint = torch.load(opt.model_name+'/model.chkpt')
    model.load_state_dict(checkpoint['model'])
all_predictions, all_targets, test_loss = test_epoch(model, test_data,opt,data['dict'],'(Testing)')
test_metrics = evals.compute_all_metrics(all_predictions,all_targets,0,opt,0,all_metrics=True)
