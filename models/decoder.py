import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import os



class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim // n_heads)

    def forward(self, query, key, value, mask=None):
        # query = key = value [batch size, sent_len, hid_dim]
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch_size, sent_len, hid_dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch_size, n_heads, sent_len_K, hid_dim / n_heads]
        # Q = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, sent_len_Q, sent_len_K]

        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, float('-inf'))
        #     attention = self.do(F.softmax(energy, dim=-1))
        #     attention = attention.masked_fill(mask == 0, 0)
        # else:
        #     attention = self.do(F.softmax(energy, dim=-1))
        energy = energy.masked_fill(mask == 0, float('-inf'))
        attention = self.do(F.softmax(energy, dim=-1))
        # attention = attention.masked_fill(mask == 0, 0)
        # attention = attention / torch.sum(attention, dim=-1, keepdim=True)

        # attention = [batch_size, n_heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch_size, n_heads, sent_len_Q, hid_dim / n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, sent_len_Q, n_heads, hid_dim / n_heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch_size, sent_len_Q, hid_dim]
        x = self.fc(x)
        # x = [batch_size, sent_len_Q, hid_dim]

        return x



class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, sent_len, hid_dim]

        x = x.permute(0, 2, 1)
        # x = [batch_size, hid_dim, sent_len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch_size, pf_dim, sent_len]
        x = self.fc_2(x)
        # x = [batch_size, hid_dim, sent_len]
        x = x.permute(0, 2, 1)
        # x = [batch_size, sent_len, hid_dim]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, hid_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound_len, atom_dim]
        # src = [batch_size, protein_len, hid_dim] # encoder output
        # trg_mask = [batch_size, compound_len]
        # src_mask = [batch_size, protein_len]
        # trg = torch.mul(trg, trg_mask.squeeze())
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg
