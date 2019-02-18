# {{*
```

## Greedy Decoding

```{.python .input}
# *}}
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
opt = torch.optim.Adam(model.parameters(), lr=2, betas=(0.9, 0.98), eps=1e-9)
schedule = NoamSchedule(opt, model.src_embed[1].d_model, 400)


for epoch in range(1):
    model.train()
    run_epoch(data_gen(V, 30, 20), model,
              SimpleLossCompute(model.generator, criterion, opt, schedule))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))
exit()

# {{*
```

> This code predicts a translation using greedy decoding for simplicity.

```{.python .input}
# *}}
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    total_prob = 0.0
    for i in range(max_len-1):
        print(ys, ys.size())
        out = model.decode(memory, src_mask,
                           ys.clone().detach(),
                           (subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        max_prob, next_word = torch.max(prob, dim = 1)
        total_prob += max_prob.data
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    print(total_prob)
    return ys

model.eval()
src = torch.tensor([[1,2,3,4,5,6,7,8,9,10]], requires_grad=False)
src_mask = torch.ones(1, 1, 10)
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
print(torch.ones(1,1,10))
# {{*
```

```{.python .input}
# *}}
def beam_decode(model, src, src_mask, max_len, start_symbol, end_symbol, k=5):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1,1).fill_(start_symbol).type_as(src.data)
    hypotheses = [(ys, 0.0)]
    for i in range(max_len):
      candidates_at_length = []
      for hypothesis, previous_prob in hypotheses:
        if hypothesis[0, -1] == end_symbol:
          candidates_at_length.append((hypothesis, previous_prob))
        else:
          # feed through model
          out = model.decode(memory, src_mask,
                               hypothesis.clone().detach(),
                               (subsequent_mask(hypothesis.size(1))
                                        .type_as(src.data)))
          probs = model.generator(out[:, -1])
          # Keep track of top k predictions for each candidates
          top_probs, predictions_at_step = torch.topk(probs, k, dim=1)
          new_hypotheses = [torch.cat([hypothesis.clone(), pred.reshape(1,1)], dim=1) for pred in predictions_at_step.flatten()]
          new_probs = top_probs.flatten().data + previous_prob
          candidates_at_length = candidates_at_length + list(zip(new_hypotheses, new_probs))
      hypotheses = sorted(candidates_at_length, key = lambda x: x[1], reverse=True)[:k]
      print(hypotheses)
    print(hypotheses[0])


model.eval()
src = torch.tensor([[1,2,3,4,5,6,7,8,9,10]], requires_grad=False)
src_mask = torch.ones(1, 1, 10)
print(beam_decode(model, src, src_mask, max_len=10, start_symbol=1, end_symbol=10))
# {{*
```

# A Real World Example

> Now we consider a real-world example using the IWSLT
German-English Translation task. This task is much smaller than the WMT task
considered in the paper, but it illustrates the whole system. We also show how
to use multi-gpu processing to make it really fast.

```{.python .input}
# *}}
# !pip install torchtext spacy
# !python -m spacy download en
# !python -m spacy download de
# {{*
```

## Data Loading
> We will load the dataset using torchtext and spacy for
tokenization.

```{.python .input}
# *}}
# For data loading.
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD,
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
# {{*
```

> Batching matters a ton for speed. We want to have very evenly divided batches,
with absolutely minimal padding. To do this we have to hack a bit around the
default torchtext batching. This code patches their default batching to make
sure we search over enough sentences to find tight batches.

## Iterators

```{.python .input}
# *}}
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
# {{*
```

```{.python .input}
# *}}
print(torch.cuda.device_count())
# {{*
```

> Now we create our model, criterion, optimizer, data iterators, and
paralelization

```{.python .input}
# *}}
# GPUs to use
devices = range(torch.cuda.device_count())
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 512
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda', 0),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda', 0),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
# {{*
```

> Now we train the model. I will play with the warmup steps a bit, but
everything else uses the default parameters.  On an AWS p3.8xlarge with 4 Tesla
V100s, this runs at ~27,000 tokens per second with a batch size of 12,000

##
Training the System

```{.python .input}
# *}}
# !wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
# {{*
```

```{.python .input}
# *}}
if True:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  SimpleLossCompute(model.generator, criterion,
                                    opt=model_opt))
        model.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                          model,
                          SimpleLossCompute(model.generator, criterion,
                          opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt")
# {{*
```

> Once trained we can decode the model to produce a set of translations. Here we
simply translate the first sentence in the validation set. This dataset is
pretty small so the translations with greedy search are reasonably accurate.

```{.python .input}
# *}}
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask,
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break
# {{*
```

# Additional Components: BPE, Search, Averaging

> So this mostly covers the
transformer model itself. There are four aspects
that we didn't cover
explicitly. We also have all these additional features
implemented in [OpenNMT-
py](https://github.com/opennmt/opennmt-py).

> 1) BPE/ Word-piece: We can use a
library to first preprocess the data into
subword units. See Rico Sennrich's
[subword-
nmt](https://github.com/rsennrich/subword-nmt) implementation. These
models will
transform the training data to look like this:

▁Die ▁Protokoll
datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen
▁bestimmte n
▁Empfänger ▁gesendet ▁werden .

> 2) Shared Embeddings: When using BPE with
shared vocabulary we can share the
same weight vectors between the source /
target / generator. See the
[(cite)](https://arxiv.org/abs/1608.05859) for
details. To add this to the model
simply do this:

```{.python .input}
# *}}
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight
# {{*
```

> 3) Beam Search: This is a bit too complicated to cover here. See the [OpenNMT-
py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/Beam.py)
for a pytorch implementation.

> 4) Model Averaging: The paper averages the last
k checkpoints to create an
ensembling effect. We can do this after the fact if
we have a bunch of models:

```{.python .input}
# *}}
def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
# {{*
```

# Results

On the WMT 2014 English-to-German translation task, the big
transformer model (Transformer (big)
in Table 2) outperforms the best previously
reported models (including ensembles) by more than 2.0
BLEU, establishing a new
state-of-the-art BLEU score of 28.4. The configuration of this model is
listed
in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our
base model
surpasses all previously published models and ensembles, at a
fraction of the training cost of any of
the competitive models.

On the WMT 2014
English-to-French translation task, our big model achieves a BLEU score of 41.0,
outperforming all of the previously published single models, at less than 1/4
the training cost of the
previous state-of-the-art model. The Transformer (big)
model trained for English-to-French used
dropout rate Pdrop = 0.1, instead of
0.3.

```{.python .input}
# *}}
Image(filename="images/results.png")
# {{*
```

> The code we have written here is a version of the base model. There are fully
trained version of this system available here  [(Example
Models)](http://opennmt.net/Models-py/).
>
> With the addtional extensions in
the last section, the OpenNMT-py replication gets to 26.9 on EN-DE WMT. Here I
have loaded in those parameters to our reimplemenation.

```{.python .input}
# *}}
# !wget https://s3.amazonaws.com/opennmt-models/en-de-model.pt
# {{*
```

```{.python .input}
# *}}
model, SRC, TGT = torch.load("en-de-model.pt")
# {{*
```

```{.python .input}
# *}}
model.eval()
sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
src = Variable(src)
src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(model, src, src_mask,
                    max_len=60, start_symbol=TGT.stoi["<s>"])
print("Translation:", end="\t")
trans = "<s> "
for i in range(1, out.size(1)):
    sym = TGT.itos[out[0, i]]
    if sym == "</s>": break
    trans += sym + " "
print(trans)
# {{*
```

## Attention Visualization

> Even with a greedy decoder the translation looks
pretty good. We can further visualize it to see what is happening at each layer
of the attention

```{.python .input}
# *}}
tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                    cbar=False, ax=ax)

for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data,
            sent, sent if h ==0 else [], ax=axs[h])
    plt.show()

for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Decoder Self Layer", layer+1)
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)],
            tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
            sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()

# {{*
```

# Conclusion

> Hopefully this code is useful for future research. Please reach
out if you have any issues. If you find this code helpful, also check out our
other OpenNMT tools.


# *}}
@inproceedings{opennmt,
  author    = {Guillaume Klein
and
               Yoon Kim and
               Yuntian Deng and
Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT:
Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi
= {10.18653/v1/P17-4012}
}


> Cheers,
> srush

{::options parse_block_html="true" /}
<div
id="disqus_thread"></div>
<script>
/**
*  RECOMMENDED CONFIGURATION VARIABLES:
EDIT AND UNCOMMENT THE SECTION BELOW
TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM
OR CMS.
*  LEARN WHY DEFINING THESE
VARIABLES IS IMPORTANT:
https://disqus.com/admin/universalcode/#configuration-
variables*/
/*
var
disqus_config = function () {
this.page.url = PAGE_URL;  //
Replace PAGE_URL
with your page's canonical URL variable
this.page.identifier =
PAGE_IDENTIFIER;
// Replace PAGE_IDENTIFIER with your page's unique identifier
variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s =
d.createElement('script');
s.src = 'https://harvard-nlp.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head ||
d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to
view the <a href="https://disqus.com/?ref_noscript">comments powered by
Disqus.</a></noscript>

<div id="disqus_thread"></div>
<script>
    /**
     *
RECOMMENDED
CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO
INSERT DYNAMIC
VALUES FROM YOUR PLATFORM OR CMS.
     *  LEARN WHY DEFINING
THESE VARIABLES IS
IMPORTANT:
https://disqus.com/admin/universalcode/#configuration-variables
*/
    /*
var
disqus_config = function () {
        this.page.url =
PAGE_URL;  // Replace
PAGE_URL with your page's canonical URL variable
this.page.identifier =
PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your
page's unique identifier
variable
    };
    */
    (function() {  // REQUIRED
CONFIGURATION VARIABLE:
EDIT THE SHORTNAME BELOW
        var d = document, s =
d.createElement('script');


        s.src =
'https://EXAMPLE.disqus.com/embed.js';  // IMPORTANT: Replace
EXAMPLE with your
forum shortname!

        s.setAttribute('data-timestamp',
+new Date());
(d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable
JavaScript to view the <a
href="https://disqus.com/?ref_noscript"
rel="nofollow">comments powered by
Disqus.</a></noscript>
