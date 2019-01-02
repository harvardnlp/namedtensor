import torch
import torch.nn as nn
import torch.nn.functional as F
from namedtensor import NamedTensor


class SimpleAttention(nn.module):
    def forward(self, query, key, value):
        return  key.contract("hidden", query) \
                   .apply(F.softmax, "keys") \
                   .contract("keys", values)


class FullAttention(nn.module):
    def __init__(self):
        self.in_projection = nn.Parameter(
            NamedTensor("key_in", "hidden"))

        self.out_projection = nn.Parameter(
            NamedTensor("hidden", "value_out"))

    #@check
    def forward(self, query, key, value, mask):
        return  key.contract("key_in", self.in_projection) \
                   .contract("hidden", query) \
                   .binop(lambda a, b: a.masked_fill(b, -1e9), mask)
                   .apply(F.softmax, "keys") \
                   .contract("keys", values) \
                   .contract("hidden", self.out_projection) \
                   .apply(F.tanh)


class MultiHeadedAttention(nn.module):
    def __init__(self):
        self.in_projection = nn.Parameter(
            NamedTensor("key_in",  "hidden"))

        self.out_projection = nn.Parameter(
            NamedTensor("hidden", "value_out"))

    @named("keys", "queries", "hidden_k", "hidden_q", "hidden_v")
    def forward(self, query, key, value):
        def in_proj(name, ):


        query = in_proj(query, "query_in") query.contract("query_in", self.in_projection)
             .shift("hidden -> (heads d_hid)", heads=8)
        key = key.contract("key_in", self.in_projection)
             .shift("hidden -> (heads d_hid)", heads=8)
        value = value.contract("query_in", self.in_projection)
             .shift("hidden -> (heads d_hid)", heads=8)

        return  key.contract("hidden", query) \
                   .binop(lambda a, b: a.masked_fill(b, -1e9), mask)
                   .apply(F.softmax, "keys") \
                   .contract("keys", values) \
                   .shift("(heads d_hid) - > hidden")
                   .contract("hidden", self.out_projection)



class Attention(nn.module):
    def __init__(self):
        pass

    def forward():
        pass
