import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from utils.image_utils import show, barplot_results, plot_img_and_bars


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
