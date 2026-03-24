# 🔤 BERT Token-Level Embedding Alignment for Sequence Labeling

A utility for extracting **token-level contextual embeddings from BERT** aligned to original word boundaries, specifically designed for **sequence labeling tasks** such as Named Entity Recognition (NER), Part-of-Speech tagging, and chunking.

This module is part of the [Hierarchical Contextualized Representation for NER](../README.md) (AAAI 2020) system, providing BERT-based contextual word embeddings as an optional feature input.

---

## Table of Contents

- [The Subword Alignment Problem](#-the-subword-alignment-problem)
- [How BERT Tokenization Works](#-how-bert-tokenization-works)
- [Alignment Strategies](#-alignment-strategies)
- [Pipeline Architecture](#-pipeline-architecture)
- [File Reference](#-file-reference)
- [Prerequisites](#-prerequisites)
- [Usage](#-usage)
- [Configuration Deep Dive](#-configuration-deep-dive)
- [Intermediate JSON Format](#-intermediate-json-format)
- [Output Format](#-output-format)
- [Integration with the NER Model](#-integration-with-the-ner-model)
- [Source Code Reference](#-source-code-reference)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

---

## 🤔 The Subword Alignment Problem

### Why This Tool Exists

BERT (Bidirectional Encoder Representations from Transformers) uses **WordPiece tokenization**, which breaks words into subword units. While this allows BERT to handle out-of-vocabulary words and rare morphological variants, it creates a fundamental mismatch for sequence labeling tasks where **each original word must receive exactly one label**.

Consider this example:

```
Original tokens:    ["John",  "Johanson",        "'s",   "house"]
BERT tokens:        ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
Token indices:       0         1       2        3       4    5    6        7
orig_to_tok_map:    [         1,      2,                4,       6              ]
```

Here, `"Johanson"` is split into two subword pieces: `"johan"` and `"##son"` (the `##` prefix indicates a continuation piece). Similarly, `"'s"` is split into `"'"` and `"s"`. BERT produces a 768-dimensional embedding for each subword piece, but we need exactly **one embedding per original word**.

### The Solution

This tool extracts BERT's internal representations and **re-aligns** them from subword-level back to the original token boundaries using configurable aggregation strategies.

---

## 🔠 How BERT Tokenization Works

BERT's tokenization is a **two-stage** process implemented in `tokenization.py`:

### Stage 1: BasicTokenizer

Performs initial text normalization and splitting:

1. **Unicode normalization** — Converts text to Unicode (UTF-8)
2. **Invalid character removal** — Strips control characters (U+0000, U+FFFD, etc.)
3. **Whitespace normalization** — Converts tabs, newlines to spaces
4. **Chinese character handling** — Adds spaces around CJK Unicode characters (U+4E00–U+9FFF, etc.)
5. **Lowercase conversion** (if `do_lower_case=True`) — Converts to lowercase and strips accent marks (Unicode category `Mn`)
6. **Punctuation splitting** — Splits on all ASCII punctuation and Unicode punctuation class (`P`)

```
Input:  "EU rejects German call"
After:  ["eu", "rejects", "german", "call"]     (with do_lower_case=True)
After:  ["EU", "rejects", "German", "call"]     (with do_lower_case=False)
```

### Stage 2: WordPieceTokenizer

Applies the **greedy longest-match-first** algorithm:

1. For each token, try to match the longest possible substring from the vocabulary
2. If a match is found starting from position > 0, prefix it with `##` (continuation marker)
3. If no match is found for any substring, replace the entire word with `[UNK]`
4. Words longer than 200 characters are directly mapped to `[UNK]`

```
Input:  "Johanson"
Step 1: Try "Johanson" → not in vocab
Step 2: Try "Johanso" → not in vocab
...
Step N: Try "Johan" → not in vocab
Step M: Try "Johan" → not in vocab → Try "johanson" (lowercase) → Try "johan" → found! → emit "johan"
        Remaining: "son" → prefix with ## → "##son" → found! → emit "##son"
Output: ["johan", "##son"]
```

### Special Tokens

BERT wraps each input sequence with special boundary tokens:

| Token | Purpose | Position |
|-------|---------|----------|
| `[CLS]` | Classification token; its embedding represents the entire sequence | First position (index 0) |
| `[SEP]` | Separator token; marks end of sequence (or boundary between sentence pairs) | Last position |
| `[UNK]` | Unknown token; replaces words that cannot be decomposed into known subwords | Variable |
| `[PAD]` | Padding token; fills remaining positions up to `max_seq_length` | After `[SEP]` |

---

## 🧩 Alignment Strategies

Three strategies are provided in `get_aligned_bert_emb.py` to map subword embeddings back to original word boundaries:

### `first` — First Subword Piece

Takes only the embedding of the **first** subword piece for each word.

```python
"Johanson" → ["johan", "##son"] → embedding("johan")
```

- ✅ **Pros**: Simplest, fastest, lowest memory usage
- ❌ **Cons**: Discards information from continuation pieces
- 📊 **Use when**: Speed is critical or words are rarely split

### `mean` — Average (Default)

Computes the **element-wise mean** across all subword piece embeddings.

```python
"Johanson" → ["johan", "##son"]
result = (embedding("johan") + embedding("##son")) / 2
```

- ✅ **Pros**: Retains information from all pieces; smooth representation
- ❌ **Cons**: Slightly slower due to accumulation
- 📊 **Use when**: Best general-purpose choice (default)

**Implementation detail** (`reduce_mean_list`):
```python
def reduce_mean_list(ls):
    """Element-wise average of multiple embedding vectors."""
    if len(ls) == 1:
        return ls[0]
    for item in ls[1:]:
        for index, value in enumerate(item):
            ls[0][index] += value
    return [value / len(ls) for value in ls[0]]
```

### `max` — Element-wise Maximum

Takes the **element-wise maximum** across all subword piece embeddings.

```python
"Johanson" → ["johan", "##son"]
result[i] = max(embedding("johan")[i], embedding("##son")[i])  # for each dimension i
```

- ✅ **Pros**: Captures the most prominent feature activation from each piece
- ❌ **Cons**: Can amplify noise; doesn't preserve average magnitude
- 📊 **Use when**: You want to capture the dominant features across subwords

### Visual Comparison

```
Word: "Johanson" → Subwords: ["johan", "##son"]

Dimension:        d1      d2      d3      d4      d5
─────────────────────────────────────────────────────
"johan"     :    0.3    -0.5     0.8     0.1    -0.2
"##son"     :    0.1     0.4    -0.3     0.7     0.6
─────────────────────────────────────────────────────
first       :    0.3    -0.5     0.8     0.1    -0.2  (take "johan")
mean        :    0.2    -0.05    0.25    0.4     0.2  (average both)
max         :    0.3     0.4     0.8     0.7     0.6  (max per dim)
```

---

## 🏗️ Pipeline Architecture

The embedding extraction is a **two-step pipeline** orchestrated by `run.sh`:

```
┌───────────────┐     ┌──────────────────────────┐     ┌────────────────────────┐
│   Input File  │     │   Step 1: extract_features│     │  Intermediate JSON     │
│ (one sentence │────▶│   .py                     │────▶│  (subword-level BERT   │
│  per line)    │     │                           │     │   embeddings + mapping)│
└───────────────┘     │  • Loads BERT checkpoint  │     └───────────┬────────────┘
                      │  • WordPiece tokenization │                 │
                      │  • Forward pass through   │                 │
                      │    Transformer layers      │                 ▼
                      │  • Extracts hidden states  │     ┌────────────────────────┐
                      │    from specified layer(s)  │     │  Step 2: get_aligned_  │
                      │  • Builds orig_to_tok_map  │     │  bert_emb.py           │
                      └──────────────────────────┘     │                        │
                                                        │  • Reads JSON output   │
                                                        │  • Applies alignment   │
                                                        │    strategy (first/    │
                                                        │    mean/max)           │
                                                        │  • Writes token-level  │
                                                        │    embeddings          │
                                                        └───────────┬────────────┘
                                                                    │
                                                                    ▼
                                                        ┌────────────────────────┐
                                                        │   Output File          │
                                                        │  (token-level aligned  │
                                                        │   embeddings, one      │
                                                        │   sentence per line,   │
                                                        │   tokens sep by |||)   │
                                                        └────────────────────────┘
```

### Internal Data Flow (Detailed)

```
Step 1 — extract_features.py:

  Input text: "EU rejects German call"
       │
       ▼
  BasicTokenizer: ["EU", "rejects", "German", "call"]   (or lowercase if do_lower_case)
       │
       ▼
  WordPieceTokenizer: ["eu", "rejects", "german", "call"]   (may produce subword splits)
       │
       ▼
  Add special tokens: ["[CLS]", "eu", "rejects", "german", "call", "[SEP]"]
       │
       ▼
  Build orig_to_tok_map: [1, 2, 3, 4]  (maps original word index → first BERT token index)
       │
       ▼
  Convert to input_ids: [101, 7327, 26192, 2446, 2655, 102, 0, 0, ...]  (padded to max_seq_length)
       │
       ▼
  BERT Forward Pass:
    ┌─────────────────────────────────────┐
    │  Token Embeddings (vocab lookup)    │
    │  + Position Embeddings (learned)    │
    │  + Token Type Embeddings (segment)  │
    │  → LayerNorm → Dropout             │
    │         ↓                           │
    │  Transformer Layer 1               │
    │    Multi-Head Self-Attention        │──▶ all_encoder_layers[0]
    │    Add & LayerNorm                  │
    │    Feed-Forward (3072 intermediate) │
    │    Add & LayerNorm                  │
    │         ↓                           │
    │  Transformer Layer 2               │──▶ all_encoder_layers[1]
    │         ↓                           │
    │       ...                           │
    │         ↓                           │
    │  Transformer Layer 12              │──▶ all_encoder_layers[11]  (= layer -1)
    └─────────────────────────────────────┘
       │
       ▼
  Extract specified layer(s) → Write JSON with per-token embeddings + orig_to_tok_map


Step 2 — get_aligned_bert_emb.py:

  Read JSON → For each sentence:
    1. Parse orig_to_tok_map (skip [CLS] at index 0)
    2. Group subword embeddings by original word boundaries
    3. Apply alignment strategy (first/mean/max)
    4. Write aligned embeddings (one sentence per line, tokens separated by |||)
```

---

## 📁 File Reference

```
BERT/
├── run.sh                  # Orchestration script — runs full two-step pipeline
├── extract_features.py     # Step 1: BERT forward pass + subword embedding extraction
│                           #   Modified from Google BERT to add orig_to_tok_map tracking
│                           #   Classes: InputExample, InputFeatures
│                           #   Key functions: convert_examples_to_features(), model_fn_builder()
├── get_aligned_bert_emb.py # Step 2: Subword → token alignment using first/mean/max
│                           #   Functions: reduce_mean_list(), reduce_max_list(), main()
├── modeling.py             # Google BERT Transformer model architecture (unmodified)
│                           #   Classes: BertConfig, BertModel
│                           #   Key arch: 12-layer Transformer with multi-head self-attention
│                           #   Activation: GELU (Gaussian Error Linear Unit)
│                           #   Normalization: LayerNorm (pre-norm variant)
├── tokenization.py         # Google BERT tokenizer (unmodified)
│                           #   Classes: FullTokenizer, BasicTokenizer, WordpieceTokenizer
│                           #   Algorithm: Greedy longest-match-first WordPiece
│                           #   Supports: Unicode, CJK characters, accent stripping
├── __init__.py             # Package marker (Apache 2.0 license header)
├── LICENSE                 # Apache 2.0 License
└── README.md               # This file
```

---

## 📋 Prerequisites

### 1. Python Environment

```bash
# Python 3.6+ recommended
python --version

# Install TensorFlow 1.x (BERT uses TF 1.x estimator API)
pip install tensorflow==1.15.5    # CPU-only
# OR
pip install tensorflow-gpu==1.15.5  # With GPU support (requires CUDA 10.0, cuDNN 7.4+)

# Other dependencies
pip install numpy
```

> **⚠️ Important**: This code uses `tf.contrib.tpu.TPUEstimator` and `tf.contrib.layers.layer_norm`, which exist only in TensorFlow 1.x. It will **not** work with TensorFlow 2.x without compatibility mode.

### 2. Download a Pre-trained BERT Model

Choose a model from [Google's BERT releases](https://github.com/google-research/bert#pre-trained-models):

| Model | Layers | Hidden | Heads | Params | Vocab | Download |
|-------|--------|--------|-------|--------|-------|----------|
| **BERT-Base Uncased** | 12 | 768 | 12 | 110M | 30,522 | [Download](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) |
| **BERT-Base Cased** | 12 | 768 | 12 | 110M | 28,996 | [Download](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) |
| **BERT-Large Uncased** | 24 | 1024 | 16 | 340M | 30,522 | [Download](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip) |
| **BERT-Large Cased** | 24 | 1024 | 16 | 340M | 28,996 | [Download](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip) |
| **BERT-Base Multilingual** | 12 | 768 | 12 | 110M | 119,547 | [Download](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) |
| **BERT-Base Chinese** | 12 | 768 | 12 | 110M | 21,128 | [Download](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) |

After downloading, unzip to get the model directory:

```
uncased_L-12_H-768_A-12/
├── bert_model.ckpt.data-00000-of-00001   # Model weights (400MB+)
├── bert_model.ckpt.index                  # Weight index
├── bert_model.ckpt.meta                   # Computation graph metadata
├── bert_config.json                       # Model architecture configuration
└── vocab.txt                              # WordPiece vocabulary (30,522 tokens for Base Uncased)
```

**What's in `bert_config.json`:**
```json
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

### 3. Prepare Input Data

Input file should have **one sentence per line**, with tokens separated by **spaces**:

```
EU rejects German call to boycott British lamb .
Peter Blackburn
BRUSSELS 1996-08-22
```

> **Note for NER**: If your data is in CoNLL column format (`word label`), extract just the words first:
> ```bash
> awk '{print $1}' train.conll | tr '\n' ' ' | sed 's/  /\n/g' > train.txt
> ```

---

## 🚀 Usage

### Quick Start

```bash
cd BERT/
bash run.sh <input_file> <output_file> <BERT_MODEL_DIR>
```

**Example:**
```bash
bash run.sh ../sample_data/eng.txt eng.train.bert /path/to/uncased_L-12_H-768_A-12
```

### What `run.sh` Does

```bash
#!/bin/bash
input_file=$1       # Your input text file
output_file=$2      # Desired output embedding file
BERT_BASE_DIR=$3    # Path to unzipped BERT model directory

layers=-6           # Which Transformer layer to extract (negative = from end)
align_strategy=mean # Alignment strategy: first | mean | max

# STEP 1: Extract raw BERT features → JSON
python extract_features.py \
    --input_file=$input_file \
    --output_file=$input_file.json \          # Intermediate JSON (will be deleted)
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --layers=$layers \                        # Which layer(s) to extract
    --max_seq_length=256 \                    # Max tokens (including [CLS]/[SEP])
    --batch_size=8 \                          # Inference batch size
    --do_lower_case=False                     # Set True for uncased models

# STEP 2: Align subword embeddings → token-level embeddings
python get_aligned_bert_emb.py \
    --input_file $input_file.json \
    --mode $align_strategy \
    --output_file $output_file

# Clean up intermediate JSON file
rm -f $input_file.json
```

---

## ⚙️ Configuration Deep Dive

### Layer Selection (`layers`)

BERT's Transformer produces different representations at each layer. Research shows different layers capture different linguistic information:

| Layer Index | Negative Index (for Base) | Linguistic Information | Recommended For |
|-------------|---------------------------|----------------------|-----------------|
| Layer 1 | `-12` | Raw token features, surface-level patterns | Morphological tasks |
| Layer 2–4 | `-11` to `-9` | Syntactic information (POS, dependency) | POS tagging, parsing |
| Layer 5–8 | `-8` to `-5` | Semantic + syntactic features | General NLP tasks |
| **Layer 7** | **`-6`** (default) | **Balanced syntactic + semantic** | **NER (recommended)** |
| Layer 9–11 | `-4` to `-2` | Task-specific features | Fine-tuned tasks |
| Layer 12 | `-1` | Most task-oriented (pre-training objective) | MLM/NSP-related tasks |

> **Multiple layers**: You can extract multiple layers simultaneously (e.g., `layers=-1,-2,-3,-4`), but only the **first listed layer** is used by `get_aligned_bert_emb.py` via `feature["layers"][0]["values"]`.

### Cased vs. Uncased (`do_lower_case`)

| Setting | Model Type | When to Use |
|---------|-----------|-------------|
| `--do_lower_case=True` | Uncased models | General NLP; case doesn't matter |
| `--do_lower_case=False` | **Cased models** | **NER** (case carries entity information: "Apple" ≠ "apple") |

> **For NER**: Always use **cased** models with `do_lower_case=False`, since capitalization is a strong signal for named entities.

### Sequence Length (`max_seq_length`)

- Includes `[CLS]` and `[SEP]` tokens (subtract 2 for actual word capacity)
- Sentences longer than this are **truncated** (words beyond the limit are lost)
- Shorter sentences are **padded** with zeros
- Maximum possible value: **512** (limited by BERT's position embeddings)
- Memory usage scales as O(n²) due to self-attention, so larger values need more GPU RAM

```
max_seq_length=256 → supports sentences up to 254 original tokens
max_seq_length=128 → supports sentences up to 126 original tokens (less memory)
max_seq_length=512 → maximum capacity, high memory usage
```

### Batch Size (`batch_size`)

| GPU Memory | Recommended Batch Size (Base) | Recommended (Large) |
|------------|------------------------------|---------------------|
| 4 GB | 4 | 1 |
| 8 GB | 8 | 2 |
| 12 GB | 16 | 4 |
| 16 GB+ | 32 | 8 |
| CPU-only | 4–8 (slow) | Not recommended |

---

## 📋 Intermediate JSON Format

Step 1 produces a JSON file (one JSON object per line) with this structure:

```json
{
  "linex_index": 0,
  "orig_to_tok_map": [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, ...],
  "features": [
    {
      "token": "[CLS]",
      "layers": [{"index": -6, "values": [0.123, -0.456, ..., 0.789]}]
    },
    {
      "token": "eu",
      "layers": [{"index": -6, "values": [0.234, -0.567, ..., 0.890]}]
    },
    {
      "token": "rejects",
      "layers": [{"index": -6, "values": [0.345, 0.678, ..., -0.901]}]
    },
    ...
    {
      "token": "[SEP]",
      "layers": [{"index": -6, "values": [-0.111, 0.222, ..., 0.333]}]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `linex_index` | int | Sentence index (0-based) |
| `orig_to_tok_map` | int[] | Maps original word position → BERT token index (padded with 0s) |
| `features` | object[] | Per-BERT-token features including the token string and layer embeddings |
| `features[i].token` | string | The BERT subword token (e.g., `"johan"`, `"##son"`) |
| `features[i].layers[j].values` | float[] | 768-dim (Base) or 1024-dim (Large) embedding vector |

---

## 📥 Output Format

The final output file has **one line per sentence**, with token embeddings separated by `|||`:

```
<emb_token1>|||<emb_token2>|||...|||<emb_tokenN>
```

Each `<emb_token>` is a **space-separated list of 768 float values** (for BERT-Base):

```
0.123456 -0.789012 ... 0.345678|||0.234567 -0.890123 ... 0.456789|||...
```

### Loading in Python

```python
import numpy as np

def load_bert_embeddings(filepath):
    """
    Load aligned BERT embeddings from the output file.
    
    Args:
        filepath: Path to the embedding file produced by the pipeline.
        
    Returns:
        List of numpy arrays, one per sentence. 
        Each array has shape (num_tokens, embedding_dim).
    """
    all_embeddings = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split("|||")
            sentence_embs = np.array([
                [float(v) for v in token.split()]
                for token in tokens
            ])
            all_embeddings.append(sentence_embs)
    return all_embeddings

# Usage
embeddings = load_bert_embeddings("eng.train.bert")
print(f"Number of sentences: {len(embeddings)}")
print(f"Sentence 1: {embeddings[0].shape}")  # e.g., (9, 768)
print(f"First token embedding (first 5 dims): {embeddings[0][0][:5]}")
```

### Verifying Alignment

Ensure token counts match between your NER data and the BERT embeddings:

```python
# Read NER data
with open("sample_data/eng.bioes.train") as f:
    sentences = []
    current = []
    for line in f:
        if line.strip():
            current.append(line.strip().split()[0])
        else:
            if current:
                sentences.append(current)
            current = []

# Compare with BERT embeddings
embeddings = load_bert_embeddings("eng.train.bert")
for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
    if len(sent) != len(emb):
        print(f"MISMATCH at sentence {i}: {len(sent)} words vs {len(emb)} embeddings")
    else:
        print(f"Sentence {i}: OK ({len(sent)} tokens)")
```

---

## 🔗 Integration with the NER Model

The BERT embeddings produced by this pipeline can be used as additional input features in the main NER system:

```
┌─────────────────────────────────────────────────────┐
│           WordSequence (main feature extractor)      │
│                                                     │
│  Word Embeddings (GloVe)──┐                         │
│  Char Features (IntNet)───┤                         │
│  Label-Similarity (LS)────┤──▶ Concatenate ──▶ BiLSTM │
│  BERT Embeddings──────────┘     (optional)          │
│  Sentence-Level Features                            │
│  Document-Level Memory Bank                         │
└─────────────────────────────────────────────────────┘
```

To enable BERT features, set the embedding path in your training config:

```ini
# In demo.train.config
bert_emb_dir=BERT/eng.train.bert
```

---

## 📜 Source Code Reference

### `extract_features.py` — Key Modifications from Google BERT

This file is based on Google's `extract_features.py` but modified to support **token-level alignment**. Key changes:

1. **`InputFeatures` class** (line 76–85): Added `orig_to_tok_map` field to track the mapping from original tokens to BERT subword tokens.

2. **`convert_examples_to_features()`** (line 207–322): 
   - Iterates over original tokens (space-split from input), not BERT tokens
   - Records `orig_to_tok_map[word_idx] = bert_token_idx` for each original word
   - Pads `orig_to_tok_map` to `max_seq_length` with zeros

3. **Output JSON** (line 410–434): Includes `orig_to_tok_map` in JSON output alongside per-token layer embeddings.

### `get_aligned_bert_emb.py` — Alignment Logic

The alignment algorithm works as follows:

```python
# For each sentence in the JSON output:
orig_to_tok_map = [id_ for id_ in data["orig_to_tok_map"] if id_ != 0] + [num_token - 1]
# This creates boundaries: orig_to_tok_map[i] = start of word i in BERT token list
# The final entry (num_token - 1) marks the end boundary

# For 'first' mode:
#   At each boundary position, take that BERT token's embedding

# For 'mean' mode:
#   Between consecutive boundaries, accumulate all subword embeddings
#   Then average them when hitting the next boundary

# For 'max' mode:
#   Between consecutive boundaries, track element-wise max
#   Then emit the max vector when hitting the next boundary
```

### `modeling.py` — BERT Architecture

The BERT model architecture consists of:

| Component | Details |
|-----------|---------|
| **Token Embeddings** | Lookup table: vocab_size × hidden_size |
| **Position Embeddings** | Learned: max_position_embeddings (512) × hidden_size |
| **Token Type Embeddings** | Learned: type_vocab_size (2) × hidden_size |
| **Transformer Encoder** | `num_hidden_layers` identical blocks, each containing: |
| &nbsp;&nbsp;→ Multi-Head Attention | `num_attention_heads` parallel attention heads, each with `hidden_size / heads` dimensions |
| &nbsp;&nbsp;→ Add & LayerNorm | Residual connection + layer normalization |
| &nbsp;&nbsp;→ Feed-Forward | Two linear layers: hidden_size → intermediate_size → hidden_size |
| &nbsp;&nbsp;→ Add & LayerNorm | Residual connection + layer normalization |
| **Activation** | GELU: `x * Φ(x)` where `Φ` is the Gaussian CDF |
| **Pooler** | Dense layer on `[CLS]` token → tanh activation |

### `tokenization.py` — Tokenizer Classes

| Class | Role |
|-------|------|
| `FullTokenizer` | End-to-end: combines BasicTokenizer + WordpieceTokenizer |
| `BasicTokenizer` | Text normalization: lowercase, accent removal, punctuation splitting, CJK handling |
| `WordpieceTokenizer` | Subword segmentation: greedy longest-match using vocab, `##` prefix for continuations |

---

## ⚠️ Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | TensorFlow not installed | `pip install tensorflow==1.15.5` |
| `ModuleNotFoundError: No module named 'modeling'` | Running from wrong directory | `cd BERT/` before running |
| `AttributeError: module 'tensorflow' has no attribute 'contrib'` | Using TensorFlow 2.x | Downgrade: `pip install tensorflow==1.15.5` |
| `ResourceExhaustedError (OOM)` | GPU memory exceeded | Reduce `batch_size` (try 4 or 2) or `max_seq_length` |
| `Mismatched embedding count` | Sentence truncated at `max_seq_length` | Increase `max_seq_length` (max 512) |
| `[UNK] token in output` | Word not in BERT vocabulary | Expected for rare words; the `[UNK]` embedding is still usable |
| `Very slow on CPU` | No GPU available | Expected — CPU inference is ~10-50x slower; use small batches |
| `FileNotFoundError: vocab.txt` | Wrong BERT model path | Verify `BERT_BASE_DIR` points to the unzipped model directory |
| `Empty output / zero embeddings` | `do_lower_case` mismatch | Use `True` for uncased models, `False` for cased models |

---

## 📊 Model Specifications

| Specification | BERT-Base | BERT-Large |
|---------------|-----------|------------|
| Transformer Layers | 12 | 24 |
| Hidden Size (embedding dim) | 768 | 1024 |
| Attention Heads | 12 (64 dims/head) | 16 (64 dims/head) |
| Feed-Forward Size | 3072 | 4096 |
| Total Parameters | ~110M | ~340M |
| Vocabulary Size | 30,522 (uncased) / 28,996 (cased) | Same |
| Max Sequence Length | 512 tokens | 512 tokens |
| Output per Token | 768-dim float vector | 1024-dim float vector |

---

## 📚 Example: Complete End-to-End Workflow

```bash
# ─── Step 1: Download BERT ───
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip

# ─── Step 2: Prepare input (one sentence per line) ───
cat > input.txt << 'EOF'
EU rejects German call to boycott British lamb .
Peter Blackburn
BRUSSELS 1996-08-22
EOF

# ─── Step 3: Run the alignment pipeline ───
cd BERT/
bash run.sh ../input.txt ../input.bert ../cased_L-12_H-768_A-12

# ─── Step 4: Verify output ───
python3 -c "
with open('../input.bert') as f:
    for i, line in enumerate(f):
        tokens = line.strip().split('|||')
        dim = len(tokens[0].split())
        print(f'Sentence {i+1}: {len(tokens)} tokens × {dim} dimensions')
"
# Expected output:
# Sentence 1: 9 tokens × 768 dimensions
# Sentence 2: 2 tokens × 768 dimensions
# Sentence 3: 2 tokens × 768 dimensions

# ─── Step 5: Use with NER training ───
cd ..
# Add to config: bert_emb_dir=input.bert
python main.py --config demo.train.config
```

---

## 🔗 References

### Papers
- **BERT**: Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). *NAACL-HLT*.
- **Transformers**: Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS*.
- **WordPiece**: Wu, Y., et al. (2016). [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144). *arXiv*.
- **GELU Activation**: Hendrycks, D. & Gimpel, K. (2016). [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415). *arXiv*.
- **Parent Project**: Luo, Y., Xiao, F., & Zhao, H. (2020). [Hierarchical Contextualized Representation for Named Entity Recognition](../README.md). *AAAI*.

### Repositories
- [Google BERT](https://github.com/google-research/bert) — Original BERT implementation
- [NCRF++](https://github.com/jiesutd/NCRFpp) — Base framework for the NER system

---

## 📄 License

The BERT source files (`extract_features.py`, `modeling.py`, `tokenization.py`) are from [Google's BERT repository](https://github.com/google-research/bert) and are licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
