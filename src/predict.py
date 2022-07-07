import torch
import torch.nn.functional as F
import numpy as np


def load_model_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


def predict(model, dataloader, label_names, weights_path="results/deepglobe/5_res/model.chkpt", n=5):
    print(f"================= PREDICT ==============")
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['model'])

    ################# predict #################
    batch = next(iter(dataloader))
    imgs, labels = batch
    print(imgs[0].shape, labels[0].shape)
    loss = 0
    for i in range(n):

        img, label = imgs[i], labels[i]
        img, label = img[None,:], label[None,:]
        print(img.shape, label.shape)
        from utils.image_utils import show, barplot_results
        show(img, index=i)
        pred, enc_output, *results = model(img)
        label = label.to(torch.float)
        normed_pred = F.sigmoid(pred).data
        # create a weighting for our inbalanced datset
        # pos_weight = torch.tensor([5.8611238, 1.21062702, 5.82371649, 9.89122553,
        #                            14.41991786, 9.75859599, 173.63953488])
        # weight_bce_loss = F.binary_cross_entropy_with_logits(pred, gold, reduction='mean', pos_weight=pos_weight)
        bce_loss = F.binary_cross_entropy_with_logits(pred, label)
        loss += bce_loss
        print(f"BCE LOSS {bce_loss.data.item()}")
        from sklearn.metrics import average_precision_score
        print(f"LOSS {loss.data.item()}")
        #print(f"macro ap {average_precision_score(label, normed_pred, average='macro')}")
        #print(f"micro ap {average_precision_score(label, normed_pred, average='micro')}")
        # print(f"micro f1 {f1_score(gold, normed_pred, average='micro')}")
        # print(f"macro f1 {f1_score(gold, normed_pred, average='macro')}")

        rounded_loss = np.round(bce_loss.item(), 2)
        barplot_results(normed_pred, label, label_names, loss=rounded_loss, index=i)
        # print(f"Shape norm pred {norm_pred.shape} end index { ((batch_idx + 1) * batch_size)}")
