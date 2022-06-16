import argparse, warnings

from data_pipeline import cars_data_loader
import utils.utils as utils
from data_pipeline.data_loader import process_data
import torch, torch.nn as nn
from lamp.Models import LAMP
from config_args import config_args, get_args
from runner import run_model
import numpy as np

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)
cars = True


def main(opt):
    # ========= Loading Dataset =========#

    #these have to be refactored
    train_data, valid_data = cars_data_loader.load_cars_dataset()

    label_adj_matrix = torch.ones(3, 3)  # full graph
    opt.tgt_vocab_size = 3  # number of labels
    opt.max_token_seq_len_d = opt.max_ar_length
    # create a random weight matric as its not loaded with glove yet
    weights_matrix = torch.FloatTensor(np.random.normal(scale=0.6, size=(3, opt.d_model)))

    # ========= Preparing Model =========#
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

    optimizer = torch.optim.Adam(model.get_trainable_parameters(), betas=(0.9, 0.98), lr=opt.lr)
    scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay,
                                                      last_epoch=-1)

    adv_optimizer = None

    # crit is not used for our training?????
    crit = utils.get_criterion(opt)

    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if torch.cuda.is_available() and opt.cuda:
        model = model.cuda()

        crit = crit.cuda()
        if opt.gpu_id != -1:
            torch.cuda.set_device(opt.gpu_id)

    if opt.load_pretrained:
        checkpoint = torch.load(opt.model_name + '/model.chkpt')
        model.load_state_dict(checkpoint['model'])

    try:
        run_model(model=model, train_data=train_data, valid_data=valid_data, crit=crit, optimizer=optimizer,
                  scheduler=scheduler, opt=opt)

    except KeyboardInterrupt:
        print('-' * 89 + '\nManual Exit')
        exit()


if __name__ == '__main__':
    main(opt)
