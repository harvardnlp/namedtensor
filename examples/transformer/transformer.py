import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from plotly.offline import iplot


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, heads=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    d = nn.Dropout(dropout)
    n = nn.LayerNorm(d_model)
    lut = nn.Embedding(src_vocab, d_model)
    s = SublayerConnection(c(n), c(d))
    attn = MultiHeadedAttention.init(heads, d_model)
    ff = PositionwiseFeedForward.init(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, c(d))

    model = EncoderDecoder(
        Encoder(clones(EncoderLayer(c(attn), c(ff), clones(c(s), 2)), N), c(n)),
        Decoder(clones(DecoderLayer(c(attn), c(attn), c(ff), clones(c(s), 3)), N), c(n)),
        nn.Sequential(Embeddings(c(lut), d_model), c(position)),
        nn.Sequential(Embeddings(c(lut), d_model), c(position)),
        Generator(nn.Linear(d_model, tgt_vocab)))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# Small example model.
tmp_model = make_model(10, 10, 2)
print(tmp_model)

class Batch:
    "Object for holding a batch of data with mask during training."
    src : torch.Tensor
    src_mask : torch.Tensor = None
    trg : torch.Tensor = None
    trg_y : torch.Tensor = None
    trg_mask : torch.Tensor = None
    pad : int = 0
    ntokens : int = 0

    def __post_init__(self):
        self.src_mask = (self.src != self.pad)
        if self.trg is not None:
            self.trg_y = self.trg[:, 1:]
            self.trg = self.trg[:, :-1]
            self.trg_mask = \
                self.make_std_mask(self.trg, self.pad)
            self.ntokens = (self.trg_y != self.pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & (
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        batch_ntokens = batch.ntokens.float()
        loss = loss_compute(out, batch.trg_y, batch_ntokens)
        # total_loss += loss
        # total_tokens += batch_ntokens
        # tokens += batch_ntokens
        # if i % 50 == 1:
        #     elapsed = time.time() - start
        #     print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
        #             (i, loss / batch_ntokens, tokens / elapsed))
        #     start = time.time()
        #     tokens = 0.0
    return total_loss / total_tokens

class LabelSmoothing(nn.Module):
    size : int
    padding_idx : int
    smoothing :float

    # def __init__(self, size, padding_idx, smoothing=0.0):
    #     super(LabelSmoothing, self).__init__()
    #     self.criterion = nn.KLDivLoss(reduction='sum')
    #     self.padding_idx = padding_idx
    #     self.confidence = 1.0 - smoothing
    #     self.smoothing = smoothing
    #     self.size = size
    #     self.true_dist = None

    def forward(self, x, target):
        crit = nn.KLDivLoss(reduction='sum')
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.long().data.unsqueeze(1), 1.0 - smoothing)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 1: # Empty tensors are dim 1 in Pytorch 0.4+
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        assert(not(true_dist.requires_grad))
        return self.criterion(x, true_dist)
# {{*
```

> Here we can see an example of how the mass is distributed to the words based
on confidence.

```{.python .input}
# *}}
#Example of label smoothing.

def label_demo():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit((predict.log()),
             (torch.LongTensor([2, 1, 0])))

    # Show the target distributions expected by the system.
    iplot([go.Heatmap(z=crit.true_dist.data)], filename='basic-heatmap')
demo(label_demo)
# {{*
```

> Label smoothing actually starts to penalize the model if it gets very
confident about a given choice.

```{.python .input}
# *}}
def label2_demo():
    crit = LabelSmoothing(5, 0, 0.2)
    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
        ])
        return crit(predict.log(),
                    torch.LongTensor([1])).data.item()
    iplot([go.Scatter(x = np.arange(1, 100), y = [loss(x) for x in range(1, 100)])])
demo(label2_demo)
# {{*
```

# A First  Example

> We can begin by trying out a simple copy-task. Given a
random set of input symbols from a small vocabulary, the goal is to generate
back those same symbols.

## Synthetic Data

```{.python .input}
# *}}
def data_gen(vocab_size, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, vocab_size,
                                                  size=(batch, 10)))
        data[:, 0] = 1
        src = data.clone().detach()
        tgt = data.clone().detach()
        assert(not(src.requires_grad))
        yield Batch(src=src, trg=tgt, pad=0)
# {{*
```

## Loss Computation

```{.python .input}
# *}}


@dataclass
class SimpleLossCompute:
    "A simple loss compute and train function."
    generator : Mod
    criterion : Mod
    opt : Mod
    schedule : Mod

    def __call__(self, x, y, norm):
        print(x.shape, y.shape)
        x = self.generator(x)
        print(x.shape, y.shape)
        einassert("btv,bt", (x, y))
        loss = self.criterion((x.contiguous().view(-1, x.size(-1))),
                              (y.float().contiguous().view(-1))) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.schedule.step()
            self.opt.zero_grad()
        return loss.data.item() * norm
