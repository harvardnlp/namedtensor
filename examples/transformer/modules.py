from namedtensor import ntorch
nn = ntorch.nn
import torch, math

class Attention(nn.Module):
    "Scaled dot product attention"
    def __init__(self, p, scale):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p)
        self.scale = scale

    def spec(self, dim_query, dim_keys):
        self.dim_query = dim_query
        self.dim_keys = dim_keys
        return self

    def forward(self, query, key, value, mask):
        scores = query.dot(self.dim_query, key) / self.scale
        scores[mask] = -1e9
        p_attn = self.dropout(scores.softmax(self.dim_keys))
        return value.dot(self.dim_keys, p_attn), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, p=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.proj = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(p, d_model)

    def spec(self, dim_keys, dim_hidden):
        self.dim_hidden = dim_hidden
        self.dim_keys = dim_keys
        self.proj.spec(dim_hidden)
        self.out_proj.spec(dim_hidden)
        self.dims_proj = ("heads", "lower")
        self.attention.spec("lower", dim_keys)
        return self

    def forward(self, query, key, value, mask):
        split = lambda x: x.split(self.dim_hidden, self.dims_proj,
                                  heads=self.d_k)
        query = split(self.proj[0](query))
        key = split(self.proj[1](key))
        value = split(self.proj[2](value))
        x, self.attn = self.attention(query, key, value, mask=mask)
        return self.out_proj(x.stack(self.dims_proj, self.dim_hidden))

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing, size, padding_idx):
        super(LabelSmoothing, self).__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.criterion = torch.nn.KLDivLoss(reduction='sum')

        # Internal
        self._off_prob = self.smoothing / (self.size - 2)
        self._on_prob = 1.0 - self.smoothing

    def spec(self, dim_batch, dim_classes):
        self.dim_classes = dim_classes
        self.dim_batch = dim_batch
        return self

    def forward(self, x, target):
        assert x.shape[self.dim_classes] == self.size
        target_dist = ntorch.tensor(x.values, names=x.dims).fill_(self._off_prob)
        target_dist[{self.dim_classes: target}] = self._on_prob
        target_dist[{self.dim_classes: self.padding_idx}] = 0
        on = {self.dim_batch: (target != self.padding_idx).nonzero()}
        return self.criterion(x[on].values, target_dist[on].values)


class Residual(nn.Module):
    def __init__(self, d_model, p):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def spec(self, dim_hidden):
        self.dim_hidden = dim_hidden
        self.norm.spec(dim_hidden)
        return self

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, p):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = ntorch.nn.Linear(input_dim, hidden_dim)
        self.w_2 = ntorch.nn.Linear(hidden_dim, output_dim)
        self.dropout = ntorch.nn.Dropout(p)

    def spec(self, dim_hidden):
        self.dim_hidden = dim_hidden
        self.w_1.spec(dim_hidden, "internal")
        self.w_2.spec("internal", dim_hidden)
        return self

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

MAX_LEN = 5000
class PositionalEmbeddings(nn.Module):
    def __init__(self, input_dim, output_dim, scale, p):
        super(PositionalEmbeddings, self).__init__()
        self.dropout = nn.Dropout(p)
        self.lut =  nn.Embedding(input_dim, output_dim)
        self.scale = scale
        self.d_model = output_dim

    def spec(self, dim_length, dim_hidden):
        self.dim_length = dim_length
        self.dim_hidden = dim_hidden
        self.pe = self.pe()
        return self

    def pe(self):
        pe = ntorch.zeros(MAX_LEN, self.d_model,
                          names=(self.dim_length, self.dim_hidden))
        position = ntorch.arange(0, MAX_LEN,
                                 names=self.dim_length).float()
        shift = ntorch.arange(0, self.d_model, 2, names=self.dim_hidden)
        div_term = ntorch.exp(shift.float() * -(math.log(10000.0) / self.d_model))
        val = ntorch.mul(position, div_term)
        pe[{self.dim_hidden: shift}] = val.sin()
        pe[{self.dim_hidden: shift + 1}] = val.cos()
        return pe

    def forward(self, x):
        x = self.lut(x) * math.sqrt(self.scale)
        x = x + self.pe[{self.dim_length: slice(0, x.shape[self.dim_length])}]
        return self.dropout(x)
