from datetime import time

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from training import test_epoch
from utils.image_utils import show, barplot_results, plot_img_and_bars, save_confusion_matrix
from utils.log import CSV_logger, calculate_metrics


def load_model_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


def predict(model, dataloader, label_names, crit, weights_path="/model.chkpt", n=32):
    print(f"================= PREDICT ==============")
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model'])

    ################# predict #################
    batch = next(iter(dataloader))
    imgs, labels = batch
    all_pred, enc_output, *results = model(imgs)
    labels = labels.to(torch.float)
    normed_pred_all = F.sigmoid(all_pred).data
    bce_loss = crit(all_pred, labels)
    rounded_loss = np.round(bce_loss.item(), 2)
    print(f"BATCH BCE LOSS {rounded_loss}")

    threshold = 0.5
    miAP = average_precision_score(labels, normed_pred_all, average='micro')
    maAP = average_precision_score(labels, normed_pred_all, average='macro')
    threshed_predictions = normed_pred_all > threshold
    miF1 = f1_score(labels, threshed_predictions, average='micro')
    maF1 = f1_score(labels, threshed_predictions, average='macro')
    print(f"BATCH macro ap {maAP}, micro ap {miAP}")
    print(f"BATCH macro F1 {maF1}, micro F1 {miF1}")


    for i in range (n):
        img, label = imgs[i][None,:], labels[i][None,:]
        pred, normed_pred = all_pred[i][None,:], normed_pred_all[i][None,:]
        bce_loss = crit(pred, label)
        rounded_loss = np.round(bce_loss.item(), 2)
        plot_img_and_bars(img, normed_pred, label, label_names, loss=rounded_loss, index=i)


def evaluate(model, dataloader, label_names, crit, opt, weights_path="/model.chkpt",):
    print(f"================= Evaluate ==============")
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model'])

    results_file_logger = CSV_logger(file_name="results", dir_name=opt.model_name)

    ################################## TEST ###################################
    start = time.time()
    all_predictions, all_targets, threshed_predictions, test_loss = test_epoch(model, dataloader, crit, opt,
                                                                               '(Testing)')
    elapsed = ((time.time() - start) / 60)
    miF1, maF1, miAP, maAP = calculate_metrics(all_targets, all_predictions, threshed_predictions)
    print('\n(Testing) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
    print(f"test_loss :  {str(test_loss)} macro ap {maAP}, micro ap {miAP}, macro F1 {maF1}, micro F1 {miF1}")

    results_file_logger.write_csv([miF1, maF1, miAP, maAP, test_loss], new_line=True)

    save_confusion_matrix(all_targets, threshed_predictions, dir_name=opt.model_name,
                          epoch=50, every_nth_epoch=50, class_names=label_names)


