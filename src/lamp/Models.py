import torch
import torch.nn as nn
from torchvision import models
from lamp.SubLayers import XavierLinear
from lamp.Encoders import RESNETEncoder
from lamp.Decoders import GraphDecoder



class LAMP(nn.Module):
    def __init__(self, n_tgt_vocab, n_max_seq_d, n_layers_dec=6, n_head=8, n_head2=8, d_word_vec=512, d_model=512,
                 d_inner_hid=1024, d_k=64, d_v=64, dec_dropout=0.1, dec_dropout2=0.1, proj_share_weight=True,
                  enc_transform='', onehot=False,
                 no_dec_self_att=False, loss='ce', label_adj_matrix=None, label_mask=None, graph_conv=False,
                 attn_type='softmax', int_preds=False, word2vec_weights: torch.FloatTensor = None, freeze_emb=False):
        """
                Main Model
        """

        super(LAMP, self).__init__()
        self.onehot = onehot
        self.loss = loss

        self.enc_vec = False
        if enc_transform != '':
            self.enc_vec = True

        ############# Encoder ###########

        self.encoder = RESNETEncoder(d_model=d_model)

        ############# Decoder ###########

        self.decoder = GraphDecoder(
            n_tgt_vocab, n_layers=n_layers_dec, n_head=n_head,
            n_head2=n_head2, d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, dropout=dec_dropout, dropout2=dec_dropout2,
            no_dec_self_att=no_dec_self_att, label_adj_matrix=label_adj_matrix,
            label_mask=label_mask, enc_vec=self.enc_vec, attn_type=attn_type, word2vec_weights=word2vec_weights,
            freeze_emb=freeze_emb)

        bias = False
        if  not proj_share_weight:
            bias = True

        assert d_model == d_word_vec


        if proj_share_weight:
            self.tgt_word_proj = XavierLinear(d_model, n_tgt_vocab, bias=bias)
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
        else:
            self.tgt_word_proj = XavierLinear(d_model, 1, bias=bias)
        if int_preds:
            self.tgt_word_proj_copy = XavierLinear(d_model, n_tgt_vocab, bias=bias)

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freezed_param_ids = set()
        if hasattr(self.encoder, 'position_enc'):
            enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
            freezed_param_ids = freezed_param_ids | enc_freezed_param_ids
        if self.onehot:
            enc_onehot_param_ids = set(map(id, self.encoder.src_word_emb.parameters()))
            freezed_param_ids = freezed_param_ids | enc_onehot_param_ids

        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, return_attns=False, int_preds=False):
        original_size = src.shape
        batch_size = original_size[0]
        # print(f"\nModel input is a tensor of size {original_size}")

        enc_output = self.encoder(src)

        src_seq = src.view(batch_size, -1) # reshape our image into a flat vector
        # apparently dec accepts sources only in a 2d shape with batchsize and pixels
        dec_output, *dec_output2 = self.decoder(src_seq, enc_output, return_attns=return_attns,
                                                int_preds=int_preds)

        seq_logit = self.tgt_word_proj(dec_output)
        seq_logit = torch.diagonal(seq_logit, 0, 1, 2)
        if int_preds:
            intermediate_preds = []
            tgt_word_proj_copy = self.tgt_word_proj.linear.weight.data.detach().repeat(batch_size, 1, 1)
            for int_idx, int_out in enumerate(dec_output2[0][:-1]):
                int_out = torch.bmm(int_out, tgt_word_proj_copy.transpose(1, 2))
                intermediate_preds += [torch.diagonal(int_out, 0, 1, 2)]
            return seq_logit.view(-1, seq_logit.size(-1)), enc_output, intermediate_preds

        elif return_attns:
            return seq_logit.view(-1, seq_logit.size(-1)), enc_output, None, dec_output2

        else:
            return seq_logit.view(-1, seq_logit.size(-1)), enc_output, None


class ResnetBaseLine(nn.Module):
    def __init__(
            self, d_model=300, pretrained=True, resnet_layers=18, freeze=False):
        super(ResnetBaseLine, self).__init__()
        if resnet_layers == 18:
            self.model = models.resnet18(pretrained=pretrained)
        elif resnet_layers == 50:
            self.model = models.resnet50(pretrained=pretrained)
        else:
            self.model = models.resnet101(pretrained=pretrained)

        # add last layer
        num_ftrs = self.model.fc.in_features  # in features
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        self.model.fc = nn.Linear(num_ftrs, d_model)  # out features are model dim

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        return self.model.parameters()

    def forward(self, img):
        x = self.model(img)
        return x, None, None #
