import argparse, warnings

import cars_data_loader
import utils.utils as utils
from utils import log
from data_pipeline.data_loader import process_data
import torch, torch.nn as nn
from lamp.Models import LAMP
from config_args import config_args, get_args
from runner import run_model, run_cars_model
import yaml
warnings.filterwarnings("ignore")

#parser = argparse.ArgumentParser()
#args = get_args(parser)
#opt = config_args(args)
cars = True


def main(opt):

    instructions = yaml.full_load(open(opt["instruction_file_path"]))
    logger = log.create_logger(instructions["log_dir"])
    lamp_config = yaml.full_load(instructions["model_instructions"])

    # ========= Loading Dataset =========#
    if not cars:
        data = torch.load(opt.data)
        print(data.keys())
        print(data["dict"].keys())
        print((data["train"].keys()))

        vocab_size = len(data['dict']['tgt']) # number of labels
        train_data, valid_data, test_data, label_adj_matrix, opt = process_data(data, opt)

    else:
        train_data, valid_data = cars_data_loader.load_cars_dataset()
        # todo get number of labels
        num_of_labels = 3
        label_adj_matrix = torch.ones(num_of_labels, num_of_labels)  # full graph
        instructions["tgt_vocab_size"] = num_of_labels  # number of labels
        instructions["max_token_seq_len_d"] = instructions["max_ar_length"]



    # ========= Preparing Model =========#
    model = LAMP(lamp_config, word2vec_weights=None)

    logger.info(level=20, msg=f"==== INSTRUCTIONS USED === ")
    logger.info(level=20, msg=instructions)
    logger.info(level=20, msg=f"==== ================= === ")

    logger.info(level=20, msg=f"==== Model Architecture === ")
    logger.info(level=20, msg=model)
    logger.info(level=20, msg=f"==== ================= === ")
    # print(opt.model_name)

    opt.total_num_parameters = int(utils.count_parameters(model))

    if opt.load_emb:
        model = utils.load_embeddings(model, '../../Data/word_embedding_dict.pth')

    optimizer = torch.optim.Adam(model.get_trainable_parameters(), betas=(0.9, 0.98), lr=opt.lr)
    scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay,
                                                      last_epoch=-1)

    adv_optimizer = None

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
        if cars:
            run_cars_model(model=model, train_data=train_data, valid_data=valid_data, crit=crit, optimizer=optimizer,
                           scheduler=scheduler, opt=opt)
        else:
            run_model(model, train_data, valid_data, test_data, crit, optimizer, adv_optimizer, scheduler, opt,
                      data['dict'])

    except KeyboardInterrupt:
        print('-' * 89 + '\nManual Exit')
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image-to-Image Translation')
    parser.add_argument(
        '--instruction',
        type=str,
        dest="instruction_file_path",
        default="../configs/basic_instructions.yaml"
    )
    parser.add_argument(
        "--override_args",
        action='append',
        type=lambda kv: {kv.split("=")[0]: kv.split("=")[1]},
        dest='override_args')

    opt = parser.parse_args().__dict__

    if opt["override_args"]:
        opt["override_args"] = dict(pair for d in opt["override_args"] for pair in d.items())
    print(opt)

    main(opt)
