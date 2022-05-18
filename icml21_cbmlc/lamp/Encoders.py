import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import lamp.Constants as Constants
from lamp.Layers import EncoderLayer,DecoderLayer
from lamp.SubLayers import ScaledDotProductAttention
from lamp.SubLayers import PositionwiseFeedForward
from lamp.SubLayers import XavierLinear
from pdb import set_trace as stop 
from lamp import utils
import copy


 
class RESNETEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1, pretrained=True):
        super(RESNETEncoder, self).__init__()

        model = models.resnet101(pretrained=pretrained)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.pooling = nn.MaxPool2d(14, 14)
        self.fc = nn.Linear(2048, d_model)    

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        x = self.features(src_seq)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x.view(src_seq.size(0),1,-1),None


class MLPEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):
        super(MLPEncoder, self).__init__()
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.fc1 = nn.Linear(n_src_vocab, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fout = nn.Linear(256, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        src_seq = src_seq.float()
        enc_output = self.dropout(F.relu(self.fc1(src_seq)))
        enc_output = self.dropout(F.relu(self.fc2(enc_output)))
        enc_output = self.dropout(F.relu(self.fc3(enc_output)))
        enc_output = self.fout(enc_output)
        return enc_output.view(src_seq.size(0),1,-1),None


class MLPEncoder1(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):
        super(MLPEncoder1, self).__init__()
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.linear1 = nn.Linear(n_src_vocab,d_model)

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        enc_output = self.linear1(src_seq)
        return enc_output.view(src_seq.size(0),1,-1),None


class MLPEncoder2(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, scale_coeff=0.01, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):
        super(MLPEncoder2, self).__init__()
        self.fc1 = nn.Linear(n_src_vocab, 256)
        self.fout = nn.Linear(256, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        enc_output = self.dropout(F.relu(self.fc1(src_seq)))
        enc_output = self.fout(enc_output)
        return enc_output.view(src_seq.size(0),1,-1),None


class MLPEncoder3(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, scale_coeff=0.01, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):
        super(MLPEncoder3, self).__init__()
        self.fc1 = nn.Linear(n_src_vocab, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fout = nn.Linear(512, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        enc_output = self.dropout(F.relu(self.fc1(src_seq)))
        enc_output = self.dropout(F.relu(self.fc2(enc_output)))
        enc_output = self.fout(enc_output)
        return enc_output.view(src_seq.size(0),1,-1),None


class GraphEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False,enc_transform='',
            dropout=0.1,no_enc_pos_embedding=False,both=False, emb=None):

        super(GraphEncoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.onehot = onehot
        self.enc_transform = enc_transform
        self.dropout = nn.Dropout(dropout) 
        self.both = both

        if onehot:
            self.src_word_emb = nn.Embedding(n_src_vocab, n_src_vocab, padding_idx=Constants.PAD)
            self.src_word_emb.weight.data.fill_(0)
            self.src_word_emb.weight.data[1:,1:] = torch.eye(self.src_word_emb.weight.data[1:].size(0))
            self.conv1 = nn.Conv1d(9, d_model, 16, stride=1, padding=8, dilation=1, groups=1, bias=True)
            self.conv2 = nn.Conv1d(d_model, d_model, 16, stride=1, padding=8, dilation=1, groups=1, bias=True)
        else:
            if emb is None:
                self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
            else:
                self.src_word_emb = emb

        if no_enc_pos_embedding is False:
            self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
            self.position_enc.weight.data = utils.position_encoding_init(n_position, d_word_vec)

        if both:
            self.logvar_layer = EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, adj, src_pos, return_attns=False):
        batch_size = src_seq.size(0)
        enc_input = self.src_word_emb(src_seq)
        
        if self.onehot:
            enc_input = F.relu(self.dropout(self.conv1(enc_input.transpose(1,2))))[:,:,0:-1]
            enc_input = F.max_pool1d(enc_input,2,2)
            enc_input = F.relu(self.conv2(enc_input).transpose(1,2))[:,0:-1,:]
            enc_input += self.position_enc(src_pos[:,0:enc_input.size(1)])
            src_seq = src_seq[:,0:enc_input.size(1)]
        elif hasattr(self, 'position_enc'):
            enc_input += self.position_enc(src_pos)

        enc_outputs = []
        
        if return_attns: enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = utils.get_attn_padding_mask(src_seq, src_seq)


        if adj:
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.float32)
            for idx in range(len(adj)):
                enc_slf_attn_mask[idx][0:adj[idx].size(0),0:adj[idx].size(0)] = utils.swap_0_1(adj[idx],1,0)
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.uint8)

        for (i, enc_layer) in enumerate(self.layer_stack):
            if self.both and i == len(self.layer_stack)-1:
                logvar_output, logvar_slf_attn = self.logvar_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
                if return_attns: enc_slf_attns += [logvar_slf_attn]
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns: enc_slf_attns += [enc_slf_attn]

        if self.enc_transform != '':
            if self.enc_transform == 'max':
                enc_output = F.max_pool1d(enc_output.transpose(1,2),x.size(1)).squeeze()
            elif self.enc_transform == 'sum':
                enc_output = enc_output.sum(1)
            elif self.enc_transform == 'mean':
                enc_output = enc_output.sum(1)/((src_seq > 0).sum(dim=1).float().view(-1,1))
            elif self.enc_transform == 'flatten':
                enc_output = enc_output.view(batch_size,-1).float()
            enc_output = enc_output.view(batch_size,1,-1)
        
        if return_attns:
            if self.both: 
                return enc_output,logvar_output,enc_slf_attns
            else:
                return enc_output,enc_slf_attns
        else:
            if self.both: 
                return enc_output,logvar_output,None
            else:
                return enc_output,None

class RNNEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):

        super(RNNEncoder, self).__init__()
        
        self.onehot = onehot

        if onehot:
            d_word_vec = 9
            self.src_word_emb = nn.Embedding(n_src_vocab, n_src_vocab, padding_idx=Constants.PAD)
            self.src_word_emb.weight.data.fill_(0)
            self.src_word_emb.weight.data[1:,1:] = torch.eye(self.src_word_emb.weight.data[1:].size(0))
            self.conv = nn.Conv1d(9, 512, 16, stride=1, padding=0, dilation=1, groups=1, bias=True)
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.brnn = nn.GRU(d_word_vec,d_model,n_layers,batch_first=True,bidirectional=True,dropout=dropout)
        self.U = nn.Linear(d_model*2,d_model)

    def forward(self, src_seq,adj, src_pos, return_attns=False):
        #print(src_seq.shape)
        #enc_input = self.src_word_emb(src_seq)
        #print(enc_input.shape)
        enc_input = torch.unsqueeze(src_seq, 2)
        enc_output,_ = self.brnn(enc_input)
        enc_output = self.U(enc_output)
        
        return enc_output,None


class FVAE(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, scale_coeff=0.01, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):
        super(FVAE, self).__init__()
        # feature layers
        self.fx1 = nn.Linear(n_src_vocab, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, d_model)
        self.fx_logvar = nn.Linear(256, d_model)

        # things they share
        self.dropout = nn.Dropout(p=dropout)
        self.scale_coeff = scale_coeff

    def forward(self, x, dummy0=None, dummy1=None, return_attns=False):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff
        logvar = self.fx_logvar(h3) * self.scale_coeff
        mu = mu.view(x.size(0),1,-1)
        logvar = logvar.view(x.size(0),1,-1)
        return mu, logvar, None 

class LVAE(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, scale_coeff=0.01, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, onehot=False, dropout=0.1):
        super(LVAE, self).__init__()
        # feature layers
        self.fx1 = nn.Linear(n_src_vocab, 512)
        self.fx2 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, d_model)
        self.fx_logvar = nn.Linear(256, d_model)

        # things they share
        self.dropout = nn.Dropout(p=dropout)
        self.scale_coeff = scale_coeff

    def forward(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        mu = self.fx_mu(h2) * self.scale_coeff
        logvar = self.fx_logvar(h2) * self.scale_coeff
        mu = mu.view(x.size(0),1,-1)
        logvar = logvar.view(x.size(0),1,-1)
        return mu, logvar, None 
