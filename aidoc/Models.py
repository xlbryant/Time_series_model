import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable
from aidoc.Modules import BottleLinear as Linear
from aidoc.Layers import EncoderLayer


def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

class Encoder(nn.Module):
    def __init__(self, n_src_diag, n_max_seq, n_layers=6, n_head=8, d_k=512, d_v=512,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        super(Encoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=0)

        self.src_visit_seq_emb = nn.Embedding(n_src_diag, d_word_vec, padding_idx=0)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src_seq, src_pos, return_attens = False):
        enc_input = self.src_visit_seq_emb(src_seq)

        enc_input += self.position_enc(src_pos)

        if return_attens:
            enc_slf_attns = []
        enc_output = enc_input
        enc_slf_attns_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attns_mask)
            if return_attens:
                enc_slf_attns += [enc_slf_attn]

        if return_attens:
            return enc_output, enc_slf_attns
        else:
            return enc_output

class Doctor(nn.Module):
    def __init__(self, n_src_diag, n_max_seq, n_layers=6, n_head=8,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, d_k = 8,
                 d_v=8, dropout=0.1, proj_share_weight=True, embs_share_weight=True, cudas=False, batch_size=64, max_length = 498):
        super(Doctor, self).__init__()
        self.cudas = cudas
        self.batch_size = batch_size
        self.max_length = max_length
        self.encoder = Encoder(n_src_diag, n_max_seq, n_layers=n_layers, n_head=n_head,
                               d_word_vec=d_word_vec, d_model=d_model,
                               d_inner_hid=d_inner_hid, dropout=dropout)

        self.alpha_fc = nn.Linear(in_features=512, out_features=1)
        init.xavier_normal(self.alpha_fc.weight)

        self.Linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512,1024),
            nn.Linear(1024,512),
            nn.Linear(512,2)
        )

        init.xavier_normal(self.Linear[1].weight)
        self.Linear[1].bias.data.zero_()
        init.xavier_normal(self.Linear[2].weight)
        self.Linear[2].bias.data.zero_()
        init.xavier_normal(self.Linear[3].weight)
        self.Linear[3].bias.data.zero_()


    def get_trainable_parameters(self):
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, label, length):

        src_seq, src_pos = src
        length = length
        labels = label
        if self.cudas:
            src_seq = src_seq.cuda()
            src_pos = src_pos.cuda()
            labels = labels.cuda()

        mask = Variable(torch.FloatTensor([[1.0 if i < length[idx] else 0 for i in range(self.max_length) ] for idx in range(src_seq.size()[0]) ]).
                        unsqueeze(2), requires_grad=False)
        if self.cudas:
            mask = mask.cuda()
        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

        enc_output =self.encoder(src_seq,src_pos)

        e = self.alpha_fc(enc_output)
        alpha = masked_softmax(e,mask)

        context = torch.bmm(torch.transpose(alpha, 1, 2), enc_output).squeeze(1)

        output = self.Linear(context)

        return output
