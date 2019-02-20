from namedtensor import ntorch

nn = ntorch.nn
from modules import *

DIM_SRC = "src"
DIM_TGT = "tgt"
DIM_HIDDEN = "hidden"
DIM_CLASS = "class"


class Params:
    def __init__(self, d_model=64, d_big=64, d_head=4, p=0.1, v=100):
        self.d_model = d_model
        self.d_big = d_big
        self.d_head = d_head
        self.p = p
        self.v = v


class EncoderDecoder(nn.Module):
    def __init__(self, params):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.src_embed = PositionalEmbeddings(
            params.v, params.d_model, params.d_model, params.p
        ).spec(DIM_SRC, DIM_HIDDEN)
        self.tgt_embed = PositionalEmbeddings(
            params.v, params.d_model, params.d_model, params.p
        ).spec(DIM_TGT, DIM_HIDDEN)
        self.generator = None

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(params) for _ in range(6)])
        self.norm = nn.LayerNorm(params.d_model).spec(DIM_HIDDEN)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            params.d_head, params.d_model, params.p
        ).spec(DIM_SRC, DIM_HIDDEN)
        self.feed_forward = PositionwiseFeedForward(
            params.d_model, params.d_model, params.d_big, params.p
        ).spec(DIM_HIDDEN)
        self.sublayer = nn.ModuleList(
            [Residual(params.d_model, params.p) for _ in range(2)]
        )
        self.sublayer.spec(DIM_HIDDEN)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, params):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(params) for _ in range(6)])
        self.norm = nn.LayerNorm(params.d_model).spec(DIM_HIDDEN)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            params.d_head, params.d_model, params.p
        ).spec(DIM_TGT, DIM_HIDDEN)
        self.src_attn = MultiHeadedAttention(
            params.d_head, params.d_model, params.p
        ).spec(DIM_SRC, DIM_HIDDEN)
        self.feed_forward = PositionwiseFeedForward(
            params.d_model, params.d_model, params.d_big, params.p
        ).spec(DIM_HIDDEN)

        self.sublayer = nn.ModuleList(
            [Residual(params.d_model, params.p) for _ in range(3)]
        )
        self.sublayer.spec(DIM_HIDDEN)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
