import time, warnings, os.path as path
import utils.evals as evals
import utils.utils as utils
import torch
from train import train_epoch
from test import test_epoch

warnings.filterwarnings("ignore")


def run_model(model, train_data, valid_data, crit, optimizer, scheduler, opt):
    logger = evals.Logger(opt)

    valid_losses = []

    losses = []

    loss_file = open(path.join(opt.model_name, 'losses.csv'), 'w+')
    for epoch_i in range(opt.epoch):
        print('================= Epoch', epoch_i + 1, '=================')
        if scheduler and opt.lr_decay > 0: scheduler.step()

        ################################## TRAIN ###################################
        start = time.time()
        all_predictions, all_targets, train_loss = train_epoch(model, train_data, crit, optimizer, opt,
                                                                    epoch=epoch_i)
        elapsed = ((time.time() - start) / 60)
        print('\n(Training) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
        train_loss = train_loss / len(train_data)
        print('B : ' + str(train_loss))

        train_metrics = evals.compute_metrics(all_predictions, all_targets, 0, opt, elapsed, all_metrics=True)

        ################################### VALID ###################################
        start = time.time()
        all_predictions, all_targets, valid_loss = test_epoch(model, valid_data, opt, '(Validation)')
        elapsed = ((time.time() - start) / 60)
        print('\n(Validation) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
        valid_loss = valid_loss / len(valid_data)
        print('B : ' + str(valid_loss))

        torch.save(all_predictions, path.join(opt.model_name, 'epochs', 'valid_preds' + str(epoch_i + 1) + '.pt'))
        torch.save(all_targets, path.join(opt.model_name, 'epochs', 'valid_targets' + str(epoch_i + 1) + '.pt'))
        valid_metrics = evals.compute_metrics(all_predictions, all_targets, 0, opt, elapsed, all_metrics=True)
        valid_losses += [valid_loss]

        best_valid, best_test = logger.evaluate_train_vali(train_metrics, valid_metrics, epoch_i,
                                                opt.total_num_parameters)
        print(opt.model_name)

        losses.append([epoch_i + 1, train_loss, valid_loss])

        if not 'test' in opt.model_name and not opt.test_only:
            utils.save_model(opt, epoch_i, model, valid_loss, valid_losses)

        loss_file.write(str(int(epoch_i + 1)))
        loss_file.write(',' + str(train_loss))
        loss_file.write(',' + str(valid_loss))
        loss_file.write('\n')
