import numpy as np
import torch
import torch.nn as nn

from lamp import utils
from lamp import DecoderLayer


# LAMP Graph decoder with initialization for wordvector
class GraphDecoder(nn.Module):
    def __init__(self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, n_head2=8, d_k=64, d_v=64, d_word_vec=512,
                 d_model=512, d_inner_hid=1024, dropout=0.1, dropout2=0.1, no_dec_self_att=False, label_adj_matrix=None,
                 label_mask=None, enc_vec=True, graph_conv=False, attn_type='softmax', word2vec_weights=None,
                 freeze_word_vector_weights=False):
        """

        :param n_tgt_vocab:
        :param n_max_seq:
        :param n_layers: number of message passing layers
        :param n_head:
        :param n_head2:
        :param d_k:
        :param d_v:
        :param d_word_vec:
        :param d_model:
        :param d_inner_hid:
        :param dropout:
        :param dropout2:
        :param no_dec_self_att:
        :param label_adj_matrix:
        :param label_mask:
        :param enc_vec:
        :param graph_conv:
        :param attn_type:
        :param word2vec_weights: Initial Edge Weights of our GNN ( From word embeddings)
        """
        super(GraphDecoder, self).__init__()
        self.enc_vec = enc_vec
        self.dropout = nn.Dropout(dropout)
        self.constant_input = torch.from_numpy(np.arange(n_tgt_vocab)).view(-1, 1)

        # here are word embedding are used ( this layer utilizes the edge weights)
        if word2vec_weights:  # we have pretrained weights
            self.tgt_word_emb = nn.Embedding.from_pretrained(embeddings=word2vec_weights,
                                                             freeze=freeze_word_vector_weights)
        else:
            self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)

        if label_adj_matrix is not None:
            for i in range(label_adj_matrix.size(0)):
                if label_adj_matrix[i].sum().item() < 1:
                    label_adj_matrix[i, i] = 1  # This prevents Nan output in attention(otherwise 0 attn weights occurs)
            self.label_mask = utils.swap_0_1(label_adj_matrix, 1, 0).unsqueeze(0)
        else:
            if label_mask == 'inveye':
                self.label_mask = 1 - torch.eye(n_tgt_vocab)
            elif label_mask == 'none':
                self.label_mask = None
            else:
                raise NotImplementedError

        self.layer_stack = nn.ModuleList()
        # we append each layer to the model
        for _ in range(n_layers):
            self.layer_stack.append(
                DecoderLayer(d_model, d_inner_hid, n_head, n_head2, d_k, d_v, dropout=dropout, dropout2=dropout2,
                             no_dec_self_att=no_dec_self_att, attn_type=attn_type))

    def forward(self, src_seq, enc_output, return_attentions=False, int_preds=False):
        """
        Forwarding in our decoder

        :param src_seq:
        :param enc_output:
        :param return_attentions:
        :param int_preds:
        :return:
        """
        batch_size = src_seq.size(0)
        int_outs, dec_slf_attns, dec_enc_attns = [], [], []

        tgt_seq = self.constant_input.repeat(1, batch_size).transpose(0, 1).cuda()

        dec_input = self.tgt_word_emb(tgt_seq)

        dec_enc_attn_pad_mask = None
        if not self.enc_vec:
            dec_enc_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, src_seq[:, 0:enc_output.size(1)])

        if self.label_mask is not None:
            dec_slf_attn_mask = self.label_mask.repeat(batch_size, 1, 1).cuda().byte()
        else:
            dec_slf_attn_mask = None

        dec_output = dec_input
        for idx, dec_layer in enumerate(self.layer_stack):
            dec_output, dec_output_int, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                               slf_attn_mask=dec_slf_attn_mask,
                                                                               dec_enc_attn_mask=dec_enc_attn_pad_mask)

            if int_preds:
                if dec_output_int is not None:
                    int_outs += [dec_output_int]
                int_outs += [dec_output]

            if return_attentions:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]

        if int_preds:
            return dec_output, int_outs
        elif return_attentions:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output, None
