import time, warnings

from torch.nn import functional as F
from tqdm import tqdm

import utils.utils as utils
import torch

from utils.image_utils import save_confusion_matrix
from utils.log import MetricTracker, calculate_metrics, CSV_logger

warnings.filterwarnings("ignore")


def run_model(model, train_data, valid_data, test_data, crit, optimizer, scheduler, opt, class_names=None):
    """
    Run training for the model using the given parameters

    """
    valid_losses = []
    results_file_logger = CSV_logger(file_name="results", dir_name=opt.model_name)

    for epoch_i in range(1, opt.epoch+1):
        print('================= Epoch', epoch_i , '=================')
        if scheduler and opt.lr_decay > 0: scheduler.step()

        ################################## TRAIN ###################################
        start = time.time()
        all_predictions, all_targets, threshed_predictions, train_loss = train_epoch(model, train_data, crit, optimizer,
                                                                                     opt, epoch_i)
        elapsed = ((time.time() - start) / 60)
        miF1, maF1, miAP, maAP = calculate_metrics(all_targets, all_predictions, threshed_predictions)
        print(f'\n(Train) elapse: {elapsed:3.3f} min')
        print(f"train_loss : {train_loss: .4f}, macro ap {maAP}, micro ap {miAP}, macro F1 {maF1}, micro F1 {miF1}")

        results_file_logger.write_csv([epoch_i, miF1, maF1, miAP, maAP, train_loss], new_line=False)

        ################################### VALID ###################################
        start = time.time()
        all_predictions, all_targets, threshed_predictions, valid_loss = test_epoch(model, valid_data, crit, opt,
                                                                                    '(Validation)')
        elapsed = ((time.time() - start) / 60)
        miF1, maF1, miAP, maAP = calculate_metrics(all_targets, all_predictions, threshed_predictions)
        print('\n(Valid) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
        print(f"valid_loss : {valid_loss: .4f}, macro ap {maAP}, micro ap {miAP}, macro F1 {maF1}, micro F1 {miF1}")
        valid_losses += [valid_loss]

        results_file_logger.write_csv([miF1, maF1, miAP, maAP, valid_loss], new_line=False)

        ################################## TEST ###################################
        start = time.time()
        all_predictions, all_targets, threshed_predictions, test_loss = test_epoch(model, test_data, crit, opt,
                                                                                   '(Testing)')
        elapsed = ((time.time() - start) / 60)
        miF1, maF1, miAP, maAP = calculate_metrics(all_targets, all_predictions, threshed_predictions)
        print('\n(Testing) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
        print(f"test_loss :  {str(test_loss)} macro ap {maAP}, micro ap {miAP}, macro F1 {maF1}, micro F1 {miF1}")

        results_file_logger.write_csv([miF1, maF1, miAP, maAP, test_loss], new_line=True)

        save_confusion_matrix(all_targets, threshed_predictions, dir_name=opt.model_name,
                              epoch=epoch_i, every_nth_epoch=10, class_names=class_names)

        utils.save_model(opt, epoch_i, model, valid_loss, valid_losses)


def train_epoch(model, train_data, crit, optimizer, opt, epoch):
    model.train()

    out_len = (opt.tgt_vocab_size)
    batch_idx, batch_size = 0, train_data.batch_size

    all_predictions = torch.zeros(len(train_data) * batch_size, out_len)
    all_targets = torch.zeros(len(train_data) * batch_size, out_len)
    loss_tracker = MetricTracker()

    threshold = 0.5
    tqdm_bar = tqdm(train_data, mininterval=0.5, desc='(Training)', leave=False)
    for batch in tqdm_bar:
        img, labels = batch
        optimizer.zero_grad()  # reset gradients
        pred, enc_output, *results = model(img)
        norm_pred = F.sigmoid(pred)  # normalize predictions to save them later NOT USED IN LOSS
        labels = labels.to(torch.float)

        # loss and backpropagation
        loss = crit(pred, labels)
        loss_tracker.update(loss.item(), batch_size)
        loss.backward()
        optimizer.step()

        ## Updates ##
        start_idx, end_idx = (batch_idx * batch_size), ((batch_idx + 1) * batch_size)
        all_predictions[start_idx:end_idx] = norm_pred.data  # save predictions
        all_targets[start_idx:end_idx] = labels.data  # save correct labels
        batch_idx += 1
        tqdm_bar.set_postfix(loss=loss_tracker.avg)

    threshed_predictions = (all_predictions > threshold).to(torch.long)

    return all_predictions, all_targets, threshed_predictions, loss_tracker.avg


def test_epoch(model, test_data, crit, opt, description):
    # setup
    model.eval()
    out_len = opt.tgt_vocab_size
    batch_size = test_data.batch_size
    all_predictions = torch.zeros(len(test_data) * batch_size, out_len)
    all_targets = torch.zeros(len(test_data) * batch_size, out_len)
    batch_idx = 0
    threshold = 0.5
    loss_tracker = MetricTracker()
    tqdm_bar = tqdm(test_data, mininterval=0.5, desc=description, leave=False)

    # evaluate
    with torch.no_grad():
        for batch in tqdm_bar:
            img, labels, = batch
            pred, enc_output, *results = model(img)
            labels = labels.to(torch.float)
            norm_pred = F.sigmoid(pred).data
            loss = crit(pred, labels)
            loss_tracker.update(loss.item(), batch_size)
            start_idx, end_idx = (batch_idx * batch_size), ((batch_idx + 1) * batch_size)
            all_predictions[start_idx:end_idx] = norm_pred
            all_targets[start_idx:end_idx] = labels
            batch_idx += 1

        threshed_predictions = (all_predictions > threshold).to(torch.long)

    return all_predictions, all_targets, threshed_predictions, loss_tracker.avg


