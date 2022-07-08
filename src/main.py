import argparse, warnings
import torch.nn.functional as F

import utils.utils as utils
import torch, torch.nn as nn

from data_pipeline.lmdb_dataloader import load_data
from data_pipeline.other_dataloaders import load_apparel_data
from lamp.Models import LAMP, ResnetBaseLine
from config_args import config_args, get_args
from runner import run_model
import numpy as np
import os
from predict import predict
warnings.filterwarnings("ignore")

from losses import AsymmetricLoss

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)
from wordembedding.glove import Glove


def main(opt):

    # Printing Debug Information
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.is_available = lambda: False
    print("Cuda is available:", torch.cuda.is_available())

    # ========= Loading Dataset =========#
    opt.max_token_seq_len_d = opt.max_ar_length
    print(f"load {opt.dataset}")
    if opt.dataset == "data/apparel":
        train_data, valid_data, test_data, labels = load_apparel_data()
    else:
        train_data, valid_data, test_data, labels = load_data(
            data_dir="data/deepglobe_patches/")
    n_output_classes =  len(labels)
    print(n_output_classes, labels)
    opt.tgt_vocab_size = n_output_classes  # number of labels
    label_adj_matrix = torch.ones(opt.tgt_vocab_size, opt.tgt_vocab_size)  # full graph
    weights_matrix = torch.FloatTensor(np.random.normal(scale=0.6, size=(opt.tgt_vocab_size, opt.d_model)))

    # ========= Preparing Model =========#
    baseline = False
    if baseline:
        print("Using Resnet Baseline")
        model = ResnetBaseLine(d_model=n_output_classes, resnet_layers=18)
    else:
        model = LAMP(opt.tgt_vocab_size, opt.max_token_seq_len_d, n_layers_dec=opt.n_layers_dec, n_head=opt.n_head,
                     n_head2=opt.n_head2, d_word_vec=opt.d_model, d_model=opt.d_model, d_inner_hid=opt.d_inner_hid,
                     d_k=opt.d_k, d_v=opt.d_v, dec_dropout=opt.dec_dropout, dec_dropout2=opt.dec_dropout2,
                     proj_share_weight=opt.proj_share_weight, encoder=opt.encoder, decoder=opt.decoder,
                     enc_transform=opt.enc_transform, onehot=opt.onehot, no_dec_self_att=opt.no_dec_self_att, loss=opt.loss,
                     label_adj_matrix=label_adj_matrix, label_mask=opt.label_mask, graph_conv=opt.graph_conv,
                     attn_type=opt.attn_type, int_preds=opt.int_preds, word2vec_weights=weights_matrix)

    print(model)
    # print(opt.model_name)

    opt.total_num_parameters = int(utils.count_parameters(model))

    if opt.load_emb:
        model = utils.load_embeddings(model, '../../Data/word_embedding_dict.pth')

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.get_trainable_parameters(),betas=(0.9, 0.999),lr=opt.lr, weight_decay=1e-5)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.get_trainable_parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum) 
    scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay,last_epoch=-1)

    adv_optimizer = None

    # crit is not used for our training, we still use BCE in the train and test loop
    if opt.loss == 'asl':
        print('using ASL')
        crit = AsymmetricLoss(gamma_neg=opt.asl_ng, gamma_pos=opt.asl_pg, clip=opt.asl_clip, eps=opt.asl_eps)
    else:
        print('using BCE ')
        crit = nn.BCEWithLogitsLoss(reduction='mean')

    ################## manage CUDA ################
    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if torch.cuda.is_available() and opt.cuda:
        model = model.cuda()
        crit = crit.cuda()
        if opt.gpu_id != -1:
            torch.cuda.set_device(opt.gpu_id)


    ######################## Load a model  ##########################
    if opt.predict == True:
        print("============== Predict ======================")
        predict(model, valid_data, labels, weights_path="results/deepglobe/enc_resnet.dec_graph.300.512.75.75.nlayers_5_4.nheads_4.proj_share.bsz_1.loss_ce.adam.lr_002.drop_10_10.priormask/model.chkpt", n=5)


    else:
        try:
            print("============== Start Training ======================")
            run_model(model=model, train_data=train_data, test_data=test_data, valid_data=valid_data, crit=crit,
                      optimizer=optimizer, scheduler=scheduler, opt=opt)

        except KeyboardInterrupt:
            print('-' * 89 + '\nManual Exit')
            exit()


if __name__ == '__main__':
    main(opt)
