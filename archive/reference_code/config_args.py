import os.path as path 
import os
import torch


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -dataset delicious -save_mode best -batch_size 32 -d_model 512 -n_layers_enc 3 -n_layers_dec 3 -n_head 4 -epoch 50 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002 -encoder 'graph' -decoder 'graph' -proj_share_weight -br_threshold 0.5 -dec_reverse -loss 'ce' -adv_lambda 1.0 -adv_type 'gan' -overwrite -thresh1 1 -int_preds -multi_gpu -test_batch_size 19

def get_args(parser):
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-dataset', type=str, default='reuters')
    parser.add_argument('-results_dir', type=str, default='results/')
    # parser.add_argument('-results_dir', type=str, default='/bigtemp/jjl5sw/deepENCODE/results/')
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-test_batch_size', type=int, default=-1)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=-1)
    parser.add_argument('-d_k', type=int, default=-1)
    parser.add_argument('-d_v', type=int, default=-1)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_head2', type=int, default=0)
    parser.add_argument('-n_layers_enc', type=int, default=5)
    parser.add_argument('-n_layers_dec', type=int, default=None)
    parser.add_argument('-optim', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-lr', type=float, default=0.0002)
    parser.add_argument('-lr_step_size', type=int, default=1)
    parser.add_argument('-lr_decay', type=float, default=0)
    parser.add_argument('-weight_decay', type=float, default=1e-5)
    parser.add_argument('-max_encoder_len', type=int, default=300)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-dec_dropout', type=float, default=-1)
    parser.add_argument('-max_ar_length', type=int, default=30)
    parser.add_argument('-label_smoothing', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-no_dec_self_att', action='store_true')
    parser.add_argument('-adj_matrix_lambda', type=float, default=0.0)
    parser.add_argument('-log', default=None)
    parser.add_argument('-loss', type=str, choices=['bce','adv','ranking', 'asl'], default='bce')
    parser.add_argument('-loss2', type=str, choices=['','l2','kl'], default='')
    parser.add_argument('-adv_lambda', type=float, default=1.0)
    parser.add_argument('-adv_type',type=str, choices=['infnet','gan'], default='gan')
    parser.add_argument('-bce_with_adv',action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-not_save', action='store_true')
    parser.add_argument('-encoder', type=str, choices=['rnn', 'graph', 'emb','mlp', 'VAE', 'VAE2', 'mlp1', 'mlp2', 'mlp3', 'resnet'], default='graph')
    parser.add_argument('-decoder', type=str, choices=['sa_m','rnn_m','sa_b','graph','mlp', 'mlp2'], default='sa_m')
    parser.add_argument('-enc_transform', type=str, choices=['max', 'mean', 'flatten','sum',''], default='')
    parser.add_argument('-lmbda', type=float, default=1)
    parser.add_argument('-label_mask', type=str, choices=['none', 'inveye', 'prior'], default='none')
    parser.add_argument('-load_emb', action='store_true')
    parser.add_argument('-attn_type', type=str, choices=['softmax', 'sigmoid'], default='softmax')
    parser.add_argument('-dual_br', type=float, default=1)
    parser.add_argument('-br_threshold', type=float, default=0.5)
    parser.add_argument('-beam_size', type=int, default=5,help='Beam size')
    parser.add_argument('-n_best', type=int, default=1)
    parser.add_argument('-onehot', action='store_true')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-pretrain', action='store_true')
    parser.add_argument('-bce_smoothing', type=float, default=1.0)
    parser.add_argument('-multi_gpu', action='store_true')
    parser.add_argument('-viz', action='store_true')
    parser.add_argument('-gpu_id', type=int, default=-1)
    parser.add_argument('-no_enc_pos_embedding', action='store_true')
    parser.add_argument('-small', action='store_true')
    parser.add_argument('-summarize_data', action='store_true')
    parser.add_argument('-overwrite', action='store_true')
    parser.add_argument('-int_preds', action='store_true')
    parser.add_argument('-test_only', action='store_true')
    parser.add_argument('-load_pretrained', action='store_true')
    parser.add_argument('-int_pred_weight', type=float, default=0.2)
    parser.add_argument('-matching_mlp', action='store_true')
    parser.add_argument('-graph_conv', action='store_true')
    parser.add_argument('-attns_loss', action='store_true')
    parser.add_argument('-thresh1', type=int, default=10)
    parser.add_argument('-name', type=str, default=None) 
    parser.add_argument('-ranking_loss', action='store_true') 
    parser.add_argument('-entropy_loss', action='store_true') 
    parser.add_argument('-spar_loss', action='store_true') 
    parser.add_argument('-ranking_weight', type=float, default=1.0)
    parser.add_argument('-ent_coeff', type=float, default=1.0)
    parser.add_argument('-spar_coeff', type=float, default=1.0)
    parser.add_argument('-spar_posratio', type=float, default=0.1)
    parser.add_argument('-spar_negratio', type=float, default=0.1)
    parser.add_argument('-spar_prob', type=float, default=0)
    parser.add_argument('-scale_coeff', type=float, default=1.0)
    parser.add_argument('-asl_pg', type=float, default=1.0)
    parser.add_argument('-asl_ng', type=float, default=4.0)
    parser.add_argument('-asl_eps', type=float, default=1e-8)
    parser.add_argument('-asl_clip', type=float, default=0.05)
    parser.add_argument('-load_src_embedding', action='store_true') 
    parser.add_argument('-load_tgt_embedding', action='store_true') 
    parser.add_argument('-tgt_emb_l2', type=float, default=0)
    parser.add_argument('-workers', type=int, default=16)
    parser.add_argument('-image_size', type=int, default=448)
    parser.add_argument('-noisy_label', type=str, default='none')
    parser.add_argument('-noisy_level', type=float, default=0)
    parser.add_argument('-freeze_emb', action='store_true') 
    opt = parser.parse_args()
    return opt




def config_args(opt):

    if torch.cuda.device_count() > 1:
        opt.multi_gpu = True
    

    # if 'reuters' in opt.dataset or 'bibtext' in opt.dataset:

    if opt.n_layers_dec is None:
        opt.n_layers_dec = opt.n_layers_enc

    if opt.dataset in ['deepsea','gm12878','gm12878_unique2','gm12878_unique','tcell']:
        opt.onehot=True

    if opt.test_batch_size <= 0:
        opt.test_batch_size = opt.batch_size

    if opt.d_v == -1:
        opt.d_v = int(opt.d_model/opt.n_head)
    if opt.d_k == -1:
        opt.d_k = int(opt.d_model/opt.n_head)

    if opt.dec_dropout == -1:
        opt.dec_dropout = opt.dropout

    if opt.dataset in ['sider'] or ('bibtext' in opt.dataset) or ('delicious' in opt.dataset) or ('bookmarks' in opt.dataset):
        opt.no_enc_pos_embedding = True
    elif opt.dataset == 'bookmarks':
        opt.max_encoder_len = 500
        opt.max_ar_length = 48

    if opt.d_inner_hid == -1:
        opt.d_inner_hid = int(opt.d_model*2)

    if opt.encoder == 'emb':
        opt.n_layers_enc = 1
    if opt.decoder in ['mlp','mlp2','rnn_m']:
        opt.n_head = 1
        opt.d_k = opt.d_model
        opt.d_k = opt.d_model


    opt.model_name = ''
    opt.model_name += 'enc_'+opt.encoder

    if opt.enc_transform != '':
        opt.model_name += '.et_'+opt.enc_transform

    opt.model_name += '.dec_'+opt.decoder
    opt.model_name += '.'+str(opt.d_model)
    opt.model_name += '.'+str(opt.d_inner_hid)
    opt.model_name += '.'+str(opt.d_k)
    opt.model_name += '.'+str(opt.d_v)
    opt.model_name += '.nlayers_'+str(opt.n_layers_enc)+'_'+str(opt.n_layers_dec)
    opt.model_name += '.nheads_'+str(opt.n_head)

    if opt.n_head2 == 0:
        opt.n_head2 = opt.n_head
    else:
        opt.model_name += '_'+str(opt.n_head2)

    if opt.decoder == 'mlp' or opt.decoder == 'mlp2':
        opt.proj_share_weight = False
    else:
        opt.proj_share_weight = True

    if opt.proj_share_weight:
        opt.model_name += '.proj_share'

    if opt.decoder == 'dual_linear':
        opt.model_name += '.dualbr_'+str(("%.2f" % opt.dual_br)).replace('.','')

    opt.model_name += '.bsz_'+str(opt.batch_size)
    opt.model_name += '.loss_'+str(opt.loss)
    if opt.loss == 'asl':
        opt.model_name += 'pg%.1f_ng%.1f_eps%f_clip%.3f' % (opt.asl_pg, opt.asl_ng, opt.asl_eps, opt.asl_clip)

    if opt.loss2 != '':
        opt.model_name += '.loss2_'+str(opt.loss2)

    if opt.loss == 'adv': 
        opt.model_name += '.'+opt.adv_type
        opt.model_name += ("%.2f" % opt.adv_lambda).replace('.','')
        opt.model_name += '.thresh1_'+str(opt.thresh1)

        if opt.bce_with_adv:
            opt.model_name += '.bce_with_adv'

    opt.model_name += '.'+str(opt.optim)
    if opt.optim == 'sgd':
        opt.model_name += '.mom_'+str(opt.momentum)

    if opt.lr < 0.0001:
        opt.model_name += '.lr_000'+str(opt.lr*1000).split('.')[1]
    else:
        opt.model_name += '.lr_'+str(opt.lr).split('.')[1]
    if opt.lr_decay > 0:
        opt.model_name += '.decay_'+str(opt.lr_decay).replace('.','')+'_'+str(opt.lr_step_size)
    if opt.weight_decay != 1e-5:
        if opt.weight_decay < 1e-4:
            opt.model_name += '.wd_000'+str(opt.weight_decay*1000).split('.')[1]
        else:
            opt.model_name += '.wd_'+str(opt.weight_decay).split('.')[1]

    opt.model_name += '.drop_'+("%.2f" % opt.dropout).split('.')[1]+'_'+("%.2f" % opt.dec_dropout).split('.')[1]


    if opt.label_smoothing > 0 and opt.decoder in ['sa_m','rnn_m']:
        opt.model_name += '.ls_'+("%.2f" % opt.label_smoothing).split('.')[1]

    if opt.decoder not in ['mlp','mlp2','graph','star','dual_linear']:
        opt.model_name += '.beam_'+str(opt.beam_size)

    if opt.decoder == 'graph' and opt.no_dec_self_att:
        opt.model_name += '.no_dec_self_att'

    if opt.decoder == 'graph' and not opt.no_dec_self_att:
        opt.model_name += '.'+opt.label_mask+'mask'

    if opt.label_mask=='random':
        opt.dec_dropout2 = 0.5
        opt.label_mask = 'none'
    else:
        opt.dec_dropout2 = False

    #if opt.dataset == 'rcv1' and opt.label_mask=='prior':
    #    opt.adj_matrix_lambda = 1

    if opt.load_emb:
        opt.model_name += '.load_emb'
    if opt.load_src_embedding:
        opt.model_name += '.load_src_emb'
    if opt.load_tgt_embedding:
        opt.model_name += '.load_tgt_emb'
        if opt.tgt_emb_l2 > 0:
            opt.model_name += '.l2_%f' % opt.tgt_emb_l2
        if opt.freeze_emb:
            opt.model_name += '-freeze'


    if opt.matching_mlp:
        opt.model_name += '.matching_mlp'

    if opt.pretrain:
        opt.model_name += '.pretrain'

    if opt.graph_conv:
        opt.model_name += '.graph_conv'

    if opt.attns_loss:
        opt.model_name += '.attns_loss'

    if opt.decoder == 'graph' and opt.int_preds:
        opt.model_name += '.int_preds_'+str(opt.int_pred_weight).replace('.','')
    else:
        opt.int_preds = False

    if opt.attn_type != 'softmax':
        opt.model_name += '.'+opt.attn_type

    if opt.bce_smoothing != 1.0:
        opt.model_name += '.bcesmoothing'

    if opt.name:
        opt.model_name = (opt.model_name+'.'+str(opt.name))

    if opt.ranking_loss:
        opt.model_name += '_ranking_loss_%.2f' % opt.ranking_weight

    if opt.entropy_loss:
        opt.model_name += '_entloss_%.2f' % opt.ent_coeff

    if opt.spar_loss:
        opt.model_name += '_sparloss_%.4f_ratio_%.4f_%.4f_partial_%.4f' % (opt.spar_coeff, opt.spar_posratio, opt.spar_negratio, opt.spar_prob)

    if opt.noisy_label != 'none':
        opt.model_name = path.join(opt.results_dir,opt.dataset+'_'+opt.noisy_label+'_'+str(opt.noisy_level),opt.model_name)
    else:
        opt.model_name = path.join(opt.results_dir,opt.dataset,opt.model_name)


    opt.data_type = opt.dataset

    opt.dataset = path.join(opt.dataroot,opt.dataset)


    opt.cuda = not opt.no_cuda
    #opt.d_word_vec = opt.d_model
    if opt.load_src_embedding or opt.load_tgt_embedding:
        opt.d_word_vec = 300
    else:
        opt.d_word_vec = opt.d_model

    if opt.small:
        opt.data = path.join(opt.dataset,'small_train_valid_test.pt')
    else:
        opt.data = path.join(opt.dataset,'train_valid_test.pt')


    if opt.decoder in ['mlp','mlp2','sa_b', 'graph','star','dual_linear']:
        opt.binary_relevance = True
    elif opt.decoder in ['sa_m','rnn_m']:
        opt.binary_relevance = False
    else:
        raise NotImplementedError
        
    print(opt.model_name)

    if (not opt.viz) and (not opt.overwrite) and (not 'test' in opt.model_name) and (path.exists(opt.model_name)) and (not opt.load_pretrained):
        overwrite_status = input('Already Exists. Overwrite? (y/n): ')
        if overwrite_status == 'rm':
            os.system('rm -rf '+opt.model_name)
        elif not 'y' in overwrite_status:
            exit(0)
    return opt
