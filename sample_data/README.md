# Sample Data

## What is here

One file: `eng.bioes.train` — **four sentences, 14 tokens**, in CoNLL column
format with BIOES tags. It exists so the pipeline and the test suite have
something to run on. It is not a training set.

```
-DOCSTART- O

EU S-ORG
reject O
German S-MISC
call O
to O
boycott O
British S-MISC
lamb O
. O

Peter B-PER
Blackburn E-PER

BRUSSELS S-LOC
1996-08-22 O
```

Tag counts: `O` ×8, `S-MISC` ×2, and one each of `S-ORG`, `B-PER`, `E-PER`,
`S-LOC`. Every label the test suite expects is present exactly once or twice,
which is enough to build all three alphabets and exercise every code path — and
far too little to learn anything. F1 numbers from a run on this file are noise.

## What is missing, and why

Three things the shipped `demo.train.config` expects are **not** in this
repository:

| Expected path | What it is | Why it is absent |
|---|---|---|
| `sample_data/origin_data/eng.bioes.{dev,test}` | CoNLL-2003 English dev and test splits | Licensed corpus; redistribution is not permitted |
| `sample_data/eng.glove` | Pretrained GloVe word vectors | Large; download separately |
| `sample_data/eng.label` | Pretrained label embeddings for the label-attention mechanism | Not published with the original code |

**CoNLL-2003** is distributed by the Reuters Corpora programme and requires a
signed agreement. The four sentences above are the standard illustrative excerpt
that appears in the task description itself; the corpus proper is not here.
Obtain it from <https://www.clips.uantwerpen.be/conll2003/ner/> and place the
splits where the config points.

**GloVe** vectors are at <https://nlp.stanford.edu/projects/glove/>. The config
defaults to `word_emb_dim=100`, matching `glove.6B.100d.txt`. The loader is
case-insensitive on the second pass, so the lowercased GloVe vocabulary still
matches capitalised entity tokens.

## Format

Space- or tab-separated columns, one token per line. The **first** column is the
token, the **last** column is the NER tag; anything between them is treated as
extra features. A blank line ends a sentence.

`-DOCSTART-` marks a document boundary in CoNLL-2003. It is read as an ordinary
token here, which matters for the memory bank: document-level context is keyed on
word position across the whole corpus rather than reset per document.

### Tag schemes

| Tag | Name | Meaning |
|---|---|---|
| `B-` | Begin | First token of a multi-token entity |
| `I-` | Inside | Middle token of a multi-token entity |
| `E-` | End | Last token of a multi-token entity |
| `S-` | Single | A one-token entity |
| `O` | Outside | Not part of an entity |

**BIOES** is what the shipped config expects. BIO also works, but is weaker: two
adjacent entities of the same type are only separable in BIO because a second
`B-` appears, whereas BIOES marks the boundary explicitly.

Convert with:

```bash
python utils/tagSchemeConverter.py BIO2BIOES train.bio train.bioes
```

`IOB2BIO`, `BIO2BIOES`, `BIOES2BIO` and `IOB2BIOES` are all available.

### Two things to check in your own data

- **End the file with a blank line.** The converters flush a sentence when they
  read one. A file that ends on a token line used to lose its final sentence
  entirely; that is now handled, but a trailing blank line is still the format
  the readers expect.
- **Every label in dev and test must appear in train.** Alphabets are frozen
  after they are built, and the label alphabet has no unknown-token fallback, so
  a label seen for the first time at evaluation raises `KeyError`.
