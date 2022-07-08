import copy
import utils.utils as utils
import torch, torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, train_data, crit, optimizer, opt, epoch):
    model.train()

    out_len = (opt.tgt_vocab_size)
    batch_idx, batch_size = 0, train_data.batch_size

    all_predictions = torch.zeros(len(train_data)*batch_size, out_len)
    all_targets = torch.zeros(len(train_data)*batch_size, out_len)

    bce_total, d_total, d_fake_total, g_total = 0, 0, 0, 0

    for batch in tqdm(train_data, mininterval=0.5, desc='(Training)', leave=False):
        loss, d_loss = 0, 0
        img, labels = batch
        #print(img.get_device(), labels.get_device())
        optimizer.zero_grad() # reset gradients
        pred, enc_output, *results = model(img)
        norm_pred = F.sigmoid(pred) # normalize predictions to save them later NOT USED IN LOSS
        labels = labels.to(torch.float)
        #
        # pos_weight = torch.tensor([5.8611238, 1.21062702, 5.82371649, 9.89122553,
        #                             14.41991786, 9.75859599, 1])#173.63953488
        # # remove unknown class
        # bce_loss = F.binary_cross_entropy_with_logits(norm_pred, labels, reduction='mean', pos_weight=pos_weight)
        #bce_loss = F.binary_cross_entropy_with_logits(pred, labels, reduction='mean')
        #with logits uses a sigmoid before calculating the loss
        bce_loss =crit(pred, labels)
        loss += bce_loss
        bce_total += bce_loss.item()


        if epoch == opt.thresh1:
            opt.init_model = copy.deepcopy(model)
        loss.backward()
        optimizer.step()
        tgt_out = labels.data
        pred_out = norm_pred.data

        ## Updates ##
        start_idx, end_idx = (batch_idx * batch_size), ((batch_idx + 1) * batch_size)
        all_predictions[start_idx:end_idx] = pred_out
        all_targets[start_idx:end_idx] = tgt_out
        batch_idx += 1

    return all_predictions, all_targets, bce_total
