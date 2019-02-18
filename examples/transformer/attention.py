class Attention(nn.Module):
    def __init__(self, p, scale):
        self.dropout = self.Dropout(p)
        self.scale = scale

    def spec(self, dim_query, dim_keys):
        self.dim_query = dim_query
        self.dim_keys = dim_keys

    def forward(self, query, key, value, scale, mask):
        "Compute 'Scaled Dot Product Attention'"
        scores = query.dot(self.dim_key, key) / scale
        scores[mask] = -1e9
        p_attn = self.dropout(scores.softmax(dim_query))
        return value.dot(self.dim_keys, p_attn), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, p=0.1):
        assert d_model % heads == 0
        d_k = d_model // heads
        proj = nn.Parameter(ntorch.zeros(3, heads, d_k, d_model))
        out_proj = nn.Parameter(torch.zeros(heads, d_k, d_model))

    def spec(self, dim_query, dim_keys, dim_hidden):
        self.dim_hidden = dim_hidden
        self.dim_query = dim_query
        self.dim_keys = dim_keys
        self.proj.spec(names=("qkv", "heads", "lower", dim_hidden))
        self.out_proj.spec(names=("heads", "lower", dim_hidden))

    def forward(self, query, key, value, mask):
        group = ntorch.stack([query, key, value], "qkv")
        query, key, value = self.proj.tensor.dot(self.dim_hidden, group).unbind("qkv")
        x, self.attn = self.attention(query, key, value, self.d_k, mask=mask)
        return self.out_proj.tensor.dot(("heads", "lower"), x)
