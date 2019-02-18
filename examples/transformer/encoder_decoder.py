class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def spec(self, dim_src_len="src", dim_tgt_len="tgt", name_hidden="hidden"):
        self.src_embed.spec(dim_src_len, name_hidden)
        self.tgt_embed.spec(dim_tgt_len, name_hidden)
        self.encoder.spec(dim_src_len, name_hidden)
        self.decoder.spec(dim_tgt_len, name_hidden)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, p):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm()

    def spec(self, dim_hidden):
        self.norm.spec(dim_hidden)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__():
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.()
        self.feed_forward = nn.Linear()
        self.sublayer =

    def spec(self, dim_hidden):
        self.feed_forward.spec(dim_hidden)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers  = nn.ModuleList
        self.norm  = Mod

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attn = Mod
        self.src_attn  = Mod
        self.feed_forward = Mod
        self.sublayer = nn.ModuleList

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_
