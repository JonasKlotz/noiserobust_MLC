import utils.utils as utils
import torch, torch.nn.functional as F
import lamp as Constants
from lamp.Translator import translate
from tqdm import tqdm


def test_epoch(model, test_data, opt, description):
    model.eval()
    out_len = opt.tgt_vocab_size
    batch_size = test_data.batch_size
    all_predictions = torch.zeros(len(test_data) * batch_size, out_len)
    all_targets = torch.zeros(len(test_data) * batch_size, out_len)
    batch_idx = 0

    bce_total = 0

    for batch in tqdm(test_data, mininterval=0.5, desc=description, leave=False):
        img, labels, = batch

        batch_loc = int(batch_idx * batch_size)

        pred, enc_output, *results = model(img)



        labels = labels.to(torch.float)
        norm_pred = F.sigmoid(pred).data
        # print(f"Shape norm pred {norm_pred.shape} end index { ((batch_idx + 1) * batch_size)}")
        # pos_weight = torch.tensor([5.8611238, 1.21062702, 5.82371649, 9.89122553,
        #                             14.41991786, 9.75859599, 1])#173.63953488])
        # bce_loss = F.binary_cross_entropy_with_logits(pred, labels, reduction='mean', pos_weight=pos_weight)
        #
        bce_loss = F.binary_cross_entropy_with_logits(pred, labels, reduction='mean')
        bce_total += bce_loss.item()

        start_idx, end_idx = (batch_idx * batch_size), ((batch_idx + 1) * batch_size)
        all_predictions[start_idx:end_idx] = norm_pred
        all_targets[start_idx:end_idx] = labels

        batch_idx += 1

    return all_predictions, all_targets, bce_total
