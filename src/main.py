# for cluster
import sys
sys.path.append("/home/users/j/jonasklotz/remotesensing")

import argparse, warnings
import torch.nn.functional as F

import utils.utils as utils
import torch, torch.nn as nn

from data_pipeline.lmdb_dataloader import load_data_from_lmdb
from data_pipeline.other_dataloaders import load_data_from_dir
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
from wordembedding.glove import load_word_embeddings


def main(opt):
    # Printing Debug Information
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.cuda.is_available = lambda: False
    print("Cuda is available:", torch.cuda.is_available())

    # ========= Loading Dataset =========#
    opt.max_token_seq_len_d = opt.max_ar_length
    print(f"load {opt.dataset}")
    if opt.dataset == "data/apparel":
        train_data, valid_data, test_data, labels = load_data_from_dir(
            data_dir='data/apparel-images-dataset', batch_size=opt.batch_ize)
    else:
        train_data, valid_data, test_data, labels = load_data_from_lmdb(
            data_dir="data/deepglobe_patches/", batch_size=opt.batch_size)
    n_output_classes = len(labels)

    opt.tgt_vocab_size = n_output_classes  # number of labels
    label_adj_matrix = torch.ones(opt.tgt_vocab_size, opt.tgt_vocab_size)  # full graph
    weights_matrix = torch.FloatTensor(np.random.normal(scale=0.6, size=(opt.tgt_vocab_size, opt.d_model)))

    # ========= Preparing Model =========#
    print(f"Using Model: {opt.model}")
    if opt.model == "resnet_base":
        model = ResnetBaseLine(d_model=n_output_classes, resnet_layers=18)
    else:
        if opt.model == "lamp":
            # load node embeddings from gauss distribution
            weights_matrix = torch.FloatTensor(np.random.normal(scale=0.6, size=(opt.tgt_vocab_size, opt.d_model)))
        else:
            # load node embeddings from glove
            try:
                weights_matrix = torch.from_numpy(
                    load_word_embeddings(data_path="data/glove", dim=opt.d_model, labels=labels)) \
                    .to(torch.float32)
            except FileNotFoundError as e:
                print(f"ERROR: Glovefile not found {e}\n"
                      f"defaulting to normal distributed weights")
                weights_matrix = torch.FloatTensor(np.random.normal(scale=0.6, size=(opt.tgt_vocab_size, opt.d_model)))

        model = LAMP(opt.tgt_vocab_size, opt.max_token_seq_len_d, n_layers_dec=opt.n_layers_dec, n_head=opt.n_head,
                     n_head2=opt.n_head2, d_word_vec=opt.d_model, d_model=opt.d_model, d_inner_hid=opt.d_inner_hid,
                     d_k=opt.d_k, d_v=opt.d_v, dec_dropout=opt.dec_dropout, dec_dropout2=opt.dec_dropout2,
                     proj_share_weight=opt.proj_share_weight,
                     enc_transform=opt.enc_transform, onehot=opt.onehot, no_dec_self_att=opt.no_dec_self_att,
                     loss=opt.loss,
                     label_adj_matrix=label_adj_matrix, label_mask=opt.label_mask, graph_conv=opt.graph_conv,
                     attn_type=opt.attn_type, int_preds=opt.int_preds, word2vec_weights=weights_matrix)

    # print(model)

    opt.total_num_parameters = int(utils.count_parameters(model))

    if opt.load_emb:
        model = utils.load_embeddings(model, '../../Data/word_embedding_dict.pth')

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.get_trainable_parameters(), betas=(0.9, 0.999), lr=opt.lr, weight_decay=1e-5)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.get_trainable_parameters(), lr=opt.lr, weight_decay=opt.weight_decay,
                                    momentum=opt.momentum)
    scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay,
                                                      last_epoch=-1)

    adv_optimizer = None

    # crit is not used for our training, we still use BCE in the train and test loop
    if opt.loss == 'asl':
        print('using ASL')
        crit = AsymmetricLoss(gamma_neg=opt.asl_ng, gamma_pos=opt.asl_pg, clip=opt.asl_clip, eps=opt.asl_eps)
    elif opt.loss == 'weighted_bce':
        print('using weighted BCE ')
        # if a dataset contains 100 positive and 300 negative examples of a single class,
        # then pos_weight for the class should be equal to 300 / 100= 3
        # The loss would act as if the dataset contains 3×100=300 positive examples.
        pos_weight = torch.tensor([5.8611238, 1.21062702, 5.82371649, 9.89122553,
                                   14.41991786, 9.75859599, 1.1])  # last elem is removed anyway
        crit = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    else:
        print("Using BCE")
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
        predict(model, valid_data, labels, crit, weights_path="results/" +
                                                              "deepglobe/deepglobe.glove_d_300.epochs_5.loss_ce.adam.lr_0001" +
                                                              "/model.chkpt", n=5)


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
