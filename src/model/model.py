import torch

from encoder import ResnetEncoder
import torch.nn as nn
from decoder import GraphDecoder
from lamp.SubLayers import XavierLinear

#class LAMPMultiLabelClassifier(pl.LightningModule):
class LAMPMultiLabelClassifier(nn.Module):
    """

    """

    def __init__(self, lamp_config: dict, word2vec_weights=None, **kwargs):
        """

        :param lamp_config: dicitonary containing model configuration
        :param kwargs:
        """
        super().__init__()
        self.config = lamp_config["Config"]
        self.encoder_config = lamp_config["Encoder"]
        self.decoder_config = lamp_config["Decoder"]

        self.onehot = self.config["onehot"]
        self.loss = self.config["loss"]
        self.enc_vec = False  # kp was das sein soll?
        if self.config["enc_transform"] != '':
            self.enc_vec = True

        self.encoder = ResnetEncoder(self.encoder_config)

        # Todo: Code -> Decoder weights(welche?) mit Glove initialisieren
        self.decoder = GraphDecoder(self.decoder_config["n_tgt_vocab"],  # was ist es?
                                    self.decoder_config["n_max_seq_d"],
                                    n_layers=self.decoder_config["n_layers_dec"],
                                    n_head=self.decoder_config["n_head"],
                                    n_head2=self.decoder_config["n_head2"],
                                    d_word_vec=self.decoder_config["d_word_vec"],
                                    d_model=self.decoder_config["d_model"],
                                    d_k=self.decoder_config["d_k"],
                                    d_v=self.decoder_config["d_v"],
                                    d_inner_hid=self.decoder_config["d_inner_hid"],
                                    dropout=self.decoder_config["dec_dropout"],
                                    dropout2=self.decoder_config["dec_dropout2"],
                                    no_dec_self_att=self.decoder_config["no_dec_self_att"],
                                    label_adj_matrix=self.decoder_config["label_adj_matrix"],
                                    label_mask=self.decoder_config["label_mask"],
                                    enc_vec=self.decoder_config["self.enc_vec"],
                                    graph_conv=self.decoder_config["graph_conv"],
                                    attn_type=self.decoder_config["attn_type"],
                                    word2vec_weights=word2vec_weights)

        # probably never used. Why wouldnt you share weights?
        bias = False
        if not self.config["proj_share_weight"]:
            bias = True

        #
        if self.config["proj_share_weight"]:
            self.tgt_word_proj = XavierLinear(d_in=self.encoder_config["d_model"],
                                              d_out=self.encoder_config["n_tgt_vocab"],
                                              bias=bias)
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
        else:
            self.tgt_word_proj = XavierLinear(self.encoder_config["d_model"],
                                              1,
                                              bias=bias)
        if self.config["int_preds"]:
            self.tgt_word_proj_copy = XavierLinear(d_in=self.encoder_config["d_model"],
                                                   d_out=self.encoder_config["n_tgt_vocab"],
                                                   bias=bias)


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

    def forward(self, src, adj, tgt_seq, binary_tgt, return_attns=False, int_preds=False):
        batch_size = src[0].size(0)
        src_seq, src_pos = src

        enc_output, *enc_self_attns = self.encoder(src_seq, adj, src_pos, return_attns=return_attns)
        dec_output, *dec_output2 = self.decoder(tgt_seq, src_seq, enc_output, return_attns=return_attns,
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
