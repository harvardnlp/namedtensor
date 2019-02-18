class LabelSmoothing(nn.Module):
    def __init__(self, smoothing, size, padding_idx):
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(reduction='sum')

        # Internal
        self._off_prob = self.smoothing / (self.size - 2)
        self._on_prob = 1.0 - self.smoothing

    def spec(self, dim_batch, dim_classes):
        self.dim_batch = dim_batch
        self.dim_classes = dim_classes
        self.criterion.spec(dim_classes)

    def forward(self, x, target):
        assert x.shape[self.dim_classes] == self.size
        target_dist = ntorch(x, names=x.dims).fill_(self._off_prob)
        target_dist[{self.dim_classes: target}] = self._on_prob
        target_dist[{self.dim_classes: self.padding_idx}] = 0
        on = {self.dim_batch: (target != self.padding_idx).nonzero(name="off")}
        return self.criterion(x[on], target_dist[on])


class Residual(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, p):
        self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout(p)

    def spec(self, dim_hidden):
        self.dim_hidden = dim_hidden
        self.norm.spec(dim_hidden)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, p):
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p)

    def spec(self, dim_hidden):
        self.dim_hidden = dim_hidden
        self.w_1.spec(dim_hidden, "internal")
        self.w_2.spec("internal", dim_hidden)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, input_dim, output_dim, scale):
        self.lut =  nn.Embedding(input_dim, output_dim)
        self.scale = scale

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.scale)

MAX_LEN = 5000
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        self.d_model = int
        self.dropout = nn.Dropout

    def spec(self, dim_length, dim_hidden):
        self.dim_length = dim_length
        self.dim_hidden = dim_hidden
        self.pe = self.pe(self.d_model)

    def pe(self):
        pe = ntorch.zeros(MAX_LEN, self.d_model,
                          names=(self.dim_length, self.dim_hidden))
        position = ntorch.arange(0, MAX_LEN,
                                 names=self.dim_length).float()
        shift = torch.arange(0, self.d_model, 2, names=self.dim_hidden)
        div_term = torch.exp(shift.float() * -(math.log(10000.0) / d_model))
        val = ntorch.mul(position, div_term)
        pe[{self.dim_hidden: shift}] = val.sin()
        pe[{self.dim_hidden: shift + 1}] = val.cos()
        return pe

    def forward(self, x):
        x = x + self.pe[{self.dim_length: slice(0, x.shape[self.dim_length])}]
        return self.dropout(x)
