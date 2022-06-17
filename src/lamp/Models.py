import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lamp.Constants as Constants
from lamp.Layers import EncoderLayer, DecoderLayer
from lamp.SubLayers import ScaledDotProductAttention
from lamp.SubLayers import PositionwiseFeedForward
from lamp.SubLayers import XavierLinear
from lamp.Encoders import RESNETEncoder
from lamp.Decoders import GraphDecoder
from pdb import set_trace as stop
from lamp import utils
import copy


class LAMP(nn.Module):
    def __init__(self, n_tgt_vocab, n_max_seq_d, n_layers_dec=6, n_head=8, n_head2=8, d_word_vec=512, d_model=512,
                 d_inner_hid=1024, d_k=64, d_v=64, dec_dropout=0.1, dec_dropout2=0.1, proj_share_weight=True,
                 encoder='selfatt', decoder='sa_m', enc_transform='', onehot=False,
                 no_dec_self_att=False, loss='ce', label_adj_matrix=None, label_mask=None, graph_conv=False,
                 attn_type='softmax', int_preds=False, word2vec_weights: torch.FloatTensor = None, freeze_emb=False):
        """

        @param n_tgt_vocab:
        @param n_max_seq_d:
        @param n_layers_dec:
        @param n_head:
        @param n_head2:
        @param d_word_vec:
        @param d_model:
        @param d_inner_hid:
        @param d_k:
        @param d_v:
        @param dec_dropout:
        @param dec_dropout2:
        @param proj_share_weight:
        @param encoder:
        @param decoder:
        @param enc_transform:
        @param onehot:
        @param no_dec_self_att:
        @param loss:
        @param label_adj_matrix: This matrix contains the graph edges in out case a full matrix
        @param label_mask:
        @param graph_conv:
        @param attn_type:
        @param int_preds:
        """

        super(LAMP, self).__init__()
        self.decoder_type = decoder
        self.onehot = onehot
        self.loss = loss

        self.enc_vec = False
        if encoder == 'mlp' or enc_transform != '':
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
        if self.decoder_type in ['graph'] and not proj_share_weight:
            bias = True

        assert d_model == d_word_vec

        if self.decoder_type != 'mlp':
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
        #print(f"\nModel input is a tensor of size {original_size}")

        src_seq = src.view(batch_size, -1) # reshape our image into a flat vector
        enc_output, *enc_self_attns = self.encoder(src_seq, original_size, return_attns=return_attns)

        # reshape src? appearantly dec accepts sources only in a2d shaoe with batchsize and pixels????
        #src1 = src.view(batch_size, -1)
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
            return seq_logit.view(-1, seq_logit.size(-1)), enc_output, enc_self_attns, dec_output2

        else:
            return seq_logit.view(-1, seq_logit.size(-1)), enc_output, None