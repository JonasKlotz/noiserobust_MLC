import copy
import utils.utils as utils
import torch, torch.nn.functional as F
from tqdm import tqdm



def train_epoch(model, train_data, crit, optimizer, opt, epoch):
    model.train()

    out_len = (opt.tgt_vocab_size) if opt.binary_relevance else (opt.tgt_vocab_size - 1)
    batch_idx, batch_size = 0, train_data.batch_size

    all_predictions = torch.zeros(len(train_data)*batch_size, out_len)
    all_targets = torch.zeros(len(train_data)*batch_size, out_len)

    bce_total, d_total, d_fake_total, g_total = 0, 0, 0, 0

    for batch in tqdm(train_data, mininterval=0.5, desc='(Training)', leave=False):
        loss, d_loss = 0, 0

        img = batch["image"]
        gold = batch["labels"]
        #print(f"Gold Shape {gold.shape}")
        # gold = tgt[:, 1:]  # result????? gold labels -> our labels
        # Non Binary Outputs
        optimizer.zero_grad()
        pred, enc_output, *results = model(img, return_attns=False, int_preds=opt.int_preds)
        norm_pred = F.sigmoid(pred)


        #gold_binary = utils.get_gold_binary(gold.data.cpu(), opt.tgt_vocab_size)  # .cuda()
        gold = gold.to(torch.float)
        pos_weight = torch.tensor([5.8611238, 1.21062702, 5.82371649, 9.89122553,
                                   14.41991786, 9.75859599, 173.63953488])
        bce_loss = F.binary_cross_entropy_with_logits(norm_pred, gold, reduction='mean', pos_weight=pos_weight)
        loss += bce_loss
        bce_total += bce_loss.item()
        if opt.int_preds and not opt.matching_mlp:
            for i in range(len(results[0])):
                bce_loss = F.binary_cross_entropy_with_logits(results[0][i], gold, reduction='mean', pos_weight=pos_weight)
                loss += (opt.int_pred_weight) * bce_loss
        if epoch == opt.thresh1:
            opt.init_model = copy.deepcopy(model)
        loss.backward()
        optimizer.step()
        tgt_out = gold.data
        pred_out = norm_pred.data

        ## Updates ##
        start_idx, end_idx = (batch_idx * batch_size), ((batch_idx + 1) * batch_size)
        all_predictions[start_idx:end_idx] = pred_out
        all_targets[start_idx:end_idx] = tgt_out
        batch_idx += 1

    return all_predictions, all_targets, bce_total
