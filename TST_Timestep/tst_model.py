import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# === Helper Functions & Classes ===
def ifnone(a, b): return b if a is None else a

def get_activation_fn(activation):
    if activation == "relu": return nn.ReLU()
    elif activation == "gelu": return nn.GELU()
    else: return activation()

def default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Module(nn.Module):
    pass

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2
    def forward(self, x): return x.transpose(self.dim1, self.dim2)

class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high
    def forward(self, x):
        return torch.sigmoid(x) * (self.high - self.low) + self.low

# === TST Core ===
class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k) / (self.d_k ** 0.5)
        if mask is not None:
            scores.masked_fill_(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return context, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int):
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.attention = _ScaledDotProductAttention(d_k)  # ✅ tambahkan ini

    def forward(self, Q, K, V, mask=None):
        bs = Q.size(0) 
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = self.attention(q_s, k_s, v_s, mask)  # ✅ gunakan instance, bukan class langsung
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)

        return self.W_O(context), attn

class _TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, dropout=0.1, activation="gelu"):
        super().__init__()
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.dropout_attn = nn.Dropout(dropout)
        self.batchnorm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), get_activation_fn(activation), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.batchnorm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))

    def forward(self, src, mask=None):
        src2, attn = self.self_attn(src, src, src, mask=mask)
        src = self.batchnorm_attn(src + self.dropout_attn(src2))
        src2 = self.ff(src)
        src = self.batchnorm_ffn(src + self.dropout_ffn(src2))
        return src

class _TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, dropout=0.1, activation='gelu', n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            _TSTEncoderLayer(q_len, d_model, n_heads, d_k, d_v, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])

    def forward(self, src):
        for mod in self.layers: src = mod(src)
        return src

class TST(nn.Module):
    def __init__(self, c_in, c_out, seq_len, max_seq_len=None, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, dropout=0.1, act="gelu", fc_dropout=0., y_range=None, verbose=False, **kwargs):
        super().__init__()
        self.c_out, self.seq_len = c_out, seq_len
        q_len = seq_len
        self.new_q_len = False
        if max_seq_len and seq_len > max_seq_len:
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(nn.ConstantPad1d(padding, 0), nn.Conv1d(c_in, d_model, kernel_size=tr_factor, padding=0, stride=tr_factor))
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs)
        else:
            self.W_P = nn.Linear(c_in, d_model)

        W_pos = torch.empty((q_len, d_model), device=default_device())
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        self.dropout = nn.Dropout(dropout)
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k, d_v, d_ff, dropout, act, n_layers)
        self.flatten = Flatten()
        self.head_nf = q_len * d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None, **kwargs):
        layers = [get_activation_fn(act), Flatten()]
        if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        layers += [nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        if self.new_q_len:
            u = self.W_P(x).transpose(2,1)
        else:
            u = self.W_P(x.transpose(2,1))
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = z.transpose(2,1).contiguous()
        return self.head(z)
