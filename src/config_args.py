import os.path as path
from datetime import datetime


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -dataset delicious -save_mode best -batch_size 32 -d_model 512 -n_layers_enc 3 -n_layers_dec 3 -n_head 4 -epoch 50 -dropout 0.2 -dec_dropout 0.2 -lr 0.0002 -encoder 'graph' -decoder 'graph' -proj_share_weight -br_threshold 0.5 -dec_reverse -loss 'ce' -adv_lambda 1.0 -adv_type 'gan' -overwrite -thresh1 1 -int_preds -multi_gpu -test_batch_size 19

def get_args(parser):
    ################  PARAMETERS WE CHANGE ##################################
    parser.add_argument('-dataset', type=str, default='deepglobe')
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-d_model', type=int, default=300) # glove vector size
    parser.add_argument('-model', type=str, choices=['CbMLC', 'resnet_base', 'lamp'], default='CbMLC')
    parser.add_argument('-lr', type=float, default=0.0002)
    parser.add_argument('-loss', type=str, choices=['asl', 'bce', 'weighted_bce'], default='bce')
    parser.add_argument('-predict', action='store_true', default=False)
    parser.add_argument('-optim', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('-dataset_path', default='data/deepglobe_patches/')
    parser.add_argument('-embedded_weights_path', default='data/glove')
    parser.add_argument('-add_noise',type=float, default=0.0)
    parser.add_argument('-sub_noise',type=float, default=0.0)

    ################ CONSTANT PARAMETERS ##################################
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-results_dir', type=str, default='results/')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-test_batch_size', type=int, default=-1)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_head2', type=int, default=0)
    parser.add_argument('-n_layers_enc', type=int, default=5)
    parser.add_argument('-n_layers_dec', type=int, default=4)
    parser.add_argument('-lr_step_size', type=int, default=1)
    parser.add_argument('-lr_decay', type=float, default=0)
    parser.add_argument('-max_encoder_len', type=int, default=300)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-max_ar_length', type=int, default=30)
    parser.add_argument('-label_smoothing', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-no_dec_self_att', action='store_true')
    parser.add_argument('-adj_matrix_lambda', type=float, default=0.0)
    parser.add_argument('-log', default=None)
    parser.add_argument('-loss2', type=str, choices=['', 'l2', 'kl'], default='')
    parser.add_argument('-adv_lambda', type=float, default=1.0)
    parser.add_argument('-adv_type', type=str, choices=['infnet', 'gan'], default='gan')
    parser.add_argument('-bce_with_adv', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-enc_transform', type=str, choices=['max', 'mean', 'flatten', 'sum', ''], default='')
    parser.add_argument('-lmbda', type=float, default=1)
    parser.add_argument('-label_mask', type=str, choices=['none', 'inveye', 'prior'], default='none')
    parser.add_argument('-load_emb', action='store_true')
    parser.add_argument('-attn_type', type=str, choices=['softmax', 'sigmoid'], default='softmax')
    parser.add_argument('-dual_br', type=float, default=1)
    parser.add_argument('-br_threshold', type=float, default=0.5)
    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-n_best', type=int, default=1)
    parser.add_argument('-onehot', action='store_true')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-pretrain', action='store_true')
    parser.add_argument('-bce_smoothing', type=float, default=1.0)
    parser.add_argument('-multi_gpu', action='store_true')
    parser.add_argument('-viz', action='store_true')
    parser.add_argument('-gpu_id', type=int, default=-1)
    parser.add_argument('-no_enc_pos_embedding', action='store_true')
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
    parser.add_argument('-asl_pg', type=float, default=1.0)
    parser.add_argument('-asl_ng', type=float, default=4.0)
    parser.add_argument('-asl_eps', type=float, default=1e-8)
    parser.add_argument('-asl_clip', type=float, default=0.05)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight_decay', type=float, default=1e-5)
    opt = parser.parse_args()
    return opt


def config_args(opt):
    opt.multi_gpu = True

    if opt.n_layers_dec is None:
        opt.n_layers_dec = opt.n_layers_enc

    if opt.test_batch_size <= 0:
        opt.test_batch_size = opt.batch_size

    opt.d_v = int(opt.d_model / opt.n_head)
    opt.d_k = int(opt.d_model / opt.n_head)

    opt.dec_dropout = opt.dropout

    opt.d_inner_hid = int(opt.d_model * 2)
    if opt.n_head2 == 0:
        opt.n_head2 = opt.n_head
    opt.proj_share_weight = True
    if opt.label_mask == 'random':
        opt.dec_dropout2 = 0.5
        opt.label_mask = 'none'
    else:
        opt.dec_dropout2 = False



    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model


    opt.binary_relevance = True

    # datetime object containing current date and time
    now = datetime.now()

    # Saving dir of the model
    opt.model_name = ''
    opt.model_name += opt.model + '/'
    opt.model_name += now.strftime("%d_%m_%H_%M_")
    opt.model_name += '.d_' + str(opt.d_model)
    opt.model_name += '.' + str(opt.loss).split('.')[1]
    opt.model_name += '.' + str(opt.optim)
    opt.model_name += '.lr_' + str(opt.lr).split('.')[1]
    opt.model_name += '.addnoise_' + str(opt.add_noise).split('.')[1]
    opt.model_name += '.subnoise_' + str(opt.sub_noise).split('.')[1]


    opt.model_name = path.join(opt.results_dir, opt.dataset, opt.model_name)
    print(f"Saving results to {opt.model_name}")

    # dataset
    opt.data_type = opt.dataset
    opt.dataset = path.join(opt.dataroot, opt.dataset)
    return opt
