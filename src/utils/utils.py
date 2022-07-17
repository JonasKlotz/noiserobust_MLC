import torch.nn as nn
import torch
import csv
import numpy as np
import lamp as Constants

debug = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(opt, epoch_i, model, valid_loss, valid_losses):
    model_state_dict = model.state_dict()
    checkpoint = {'model': model_state_dict, 'settings': opt, 'epoch': epoch_i}
    if opt.save_mode == 'all':
        model_name = opt.model_name + '/accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_loss)
        torch.save(checkpoint, model_name)
    elif opt.save_mode == 'best':
        model_name = opt.model_name + '/model.chkpt'
        try:
            if valid_loss >= min(valid_losses):
                torch.save(checkpoint, model_name)
                print('[Info] The checkpoint file has been updated.')
        except:
            pass


def dprint(*string):
    if debug:
        print(*string)
