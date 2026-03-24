# 🧠 Model Architecture — Hierarchical Contextualized NER

This directory contains the **neural network model components** for the Hierarchical Contextualized Representation for Named Entity Recognition system. The model implements a four-level hierarchical feature extraction pipeline followed by structured sequence decoding.

---

## Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Module Dependency Graph](#-module-dependency-graph)
- [File Reference](#-file-reference)
- [SeqLabel — Top-Level Model](#-seqlabel--top-level-model)
- [WordSequence — Hierarchical Feature Extractor](#-wordsequence--hierarchical-feature-extractor)
- [WordRep — Word Representation](#-wordrep--word-representation)
- [IntNet — Character-Level CNN](#-intnet--character-level-inception-cnn)
- [SentenceRep — Sentence-Level Features](#-sentencerep--sentence-level-feature-extractor)
- [MemoryBank — Document-Level Context](#-memorybank--document-level-memory-network)
- [CRF — Structured Sequence Decoding](#-crf--conditional-random-field)
- [Data Flow](#-data-flow)
- [Tensor Shapes Reference](#-tensor-shapes-reference)
- [Hyperparameter Impact](#-hyperparameter-impact)

---

## 🏗️ Architecture Overview

The model builds representations at **four hierarchical levels**, each capturing context at a different granularity:

```
Level 4: Document    ┌─────────────────────────────────────────────────────────┐
                     │   MemoryBank — cross-sentence context via attention     │
                     │   over stored hidden states of co-occurring words       │
                     └──────────────────────┬──────────────────────────────────┘
                                            │ interpolation: (1-α)·word + α·doc
                                            ▼
Level 3: Word Seq    ┌─────────────────────────────────────────────────────────┐
                     │   BiLSTM — processes word+sentence features             │
                     │   hidden2tag — projects to label emission scores         │
                     └──────────────────────┬──────────────────────────────────┘
                                            │
         ┌──────────────────────────────────┤
         │                                  │
Level 2: Sentence    ┌──────────────────┐   │
                     │   SentenceRep    │   │
                     │   (separate      │   │
                     │    BiLSTM/CNN)   │   │
                     │   + label-       │   │
                     │     attention    │   │
                     └────────┬─────────┘   │
                              │ concat      │
                              ▼             │
Level 1: Word        ┌─────────────────────────────────────────────────────────┐
                     │   WordRep                                               │
                     │   ┌───────────┐  ┌─────────────┐  ┌──────────────────┐ │
                     │   │ Word Emb  │  │   IntNet    │  │ Label-Similarity │ │
                     │   │ (GloVe)   │  │ (Char CNN)  │  │  (Cosine Sim)   │ │
                     │   └───────────┘  └─────────────┘  └──────────────────┘ │
                     └─────────────────────────────────────────────────────────┘
                                            │
                                            ▼
         Decoding    ┌─────────────────────────────────────────────────────────┐
                     │   CRF (Viterbi decoding + NLL loss)                     │
                     │   OR Softmax (argmax decoding + NLLLoss)                │
                     └─────────────────────────────────────────────────────────┘
```

---

## 🔗 Module Dependency Graph

```
seqlabel.py (SeqLabel)
├── wordsequence.py (WordSequence)
│   ├── wordrep.py (WordRep)
│   │   └── IntNet.py (IntNet) ............. Character-level CNN
│   ├── SentenceRep.py (SentenceRep)
│   │   └── wordrep.py (WordRep) ........... Separate instance for sentence features
│   │       └── IntNet.py (IntNet)
│   └── MemoryBank.py (MemoryBank) ......... Document-level memory network
└── crf.py (CRF) ........................... Structured sequence decoding
```

> **Note**: `SentenceRep` instantiates its own `WordRep` (with `sentence_level=True` to disable label-similarity and avoid circular dependency). This means the model has **two separate WordRep instances** with separate parameters.

---

## 📁 File Reference

| File | Class | Lines | Purpose |
|------|-------|-------|---------|
| `__init__.py` | — | 2 | Python package marker |
| `seqlabel.py` | `SeqLabel` | ~130 | Top-level model: combines feature extraction + CRF/softmax loss |
| `wordsequence.py` | `WordSequence` | ~265 | Core hierarchical feature extractor (word + sentence + document) |
| `wordrep.py` | `WordRep` | ~120 | Word embeddings + IntNet char features + label-similarity scores |
| `IntNet.py` | `IntNet` | ~140 | Inception-style multi-scale character CNN (kernel 3 + kernel 5) |
| `SentenceRep.py` | `SentenceRep` | ~130 | Sentence-level global feature extraction (BiLSTM/GRU/CNN) |
| `MemoryBank.py` | `MemoryBank` | ~130 | Document-level memory network with cosine attention retrieval |
| `crf.py` | `CRF` | ~488 | Conditional Random Field: forward algorithm, Viterbi, n-best decoding |

---

## 🏷️ SeqLabel — Top-Level Model

**File**: `seqlabel.py` | **Class**: `SeqLabel(nn.Module)`

The entry point for the entire model. Orchestrates the feature extraction pipeline and loss computation.

### Responsibilities

1. **Training** (`calculate_loss`): Runs the full forward pass through `WordSequence`, computes loss using CRF or softmax, and returns both the loss and predicted tag sequence.
2. **Inference** (`forward`): Runs feature extraction and decoding without loss computation.

### Key Design Decision

The `label_alphabet_size` is incremented by **+2** before building `WordSequence`. This reserves two additional tag indices for the CRF's internal `START_TAG` and `STOP_TAG` markers, which are never predicted but are essential for the CRF's transition matrix boundary conditions.

### API

```python
# Training
loss, tag_seq = model.calculate_loss(
    word_inputs,       # (batch, sent_len) — word indices
    word_seq_lengths,  # (batch,) — actual lengths
    char_inputs,       # (batch*sent_len, word_len) — char indices
    char_seq_lengths,  # (batch*sent_len,) — char lengths
    char_seq_recover,  # Recovery indices for char order after sorting
    batch_label,       # (batch, sent_len) — gold labels
    mask,              # (batch, sent_len) — binary mask (1=valid, 0=pad)
    batch_idx          # (batch, sent_len) — global word instance indices
)

# Inference
tag_seq = model(word_inputs, word_seq_lengths, char_inputs,
                char_seq_lengths, char_seq_recover, mask, batch_idx)
```

### Loss Computation

| Mode | Loss Function | Decoding |
|------|--------------|----------|
| `use_crf=True` | CRF negative log-likelihood (structured loss over entire sequence) | Viterbi (dynamic programming) |
| `use_crf=False` | `NLLLoss` with `log_softmax` (independent per-position loss) | Argmax per position |

---

## ⚡ WordSequence — Hierarchical Feature Extractor

**File**: `wordsequence.py` | **Class**: `WordSequence(nn.Module)`

The core of the model. Integrates all four levels of hierarchical representation.

### Forward Pass Pipeline

```python
def forward(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
            char_seq_recover, mask, idx_inputs):

    # Step 1: Word-level (WordRep)
    #   → word_emb + char_features + label_similarity_scores
    word_represent, label_embs, word_embs = self.wordrep(...)

    # Step 2: Sentence-level (SentenceRep + label attention)
    #   → global sentence context for each word position
    sentence_represent = self.get_sentence_embedd(...)

    # Step 3: Concatenate word + sentence features
    word_represent = concat([word_represent, sentence_represent])

    # Step 4: BiLSTM over combined features
    #   → packed sequence → LSTM → unpack → dropout
    feature_out = BiLSTM(word_represent)

    # Step 5: Project to label space
    outputs = hidden2tag(feature_out)  # (batch, sent_len, num_labels)

    # Step 6: Document-level (MemoryBank)
    doc_represent = self.get_document_embedd(...)

    # Step 7: Interpolate word and document predictions
    outputs = (1 - α) * outputs + α * doc_represent

    return outputs  # emission scores → consumed by CRF
```

### Label-Attention Mechanism (`get_sentence_embedd`)

This is a key innovation of the paper. It uses label-similarity scores as attention weights to create a label-aware sentence representation:

```
1. SentenceRep extracts global features:     (batch, sent_len, global_hidden_dim)
2. Refine label-similarity via 1D CNN:       (batch, sent_len, num_labels)
3. Max-pool over label dimension:            (batch, sent_len)
4. Masked softmax → attention weights:       (batch, sent_len)
5. Weighted sum of sentence features:        (batch, 1, global_hidden_dim)
6. Repeat for all word positions:            (batch, sent_len, global_hidden_dim)
```

### Memory Bank Interpolation (`get_document_embedd`)

The document-level context is blended with word-level predictions via a hyperparameter α (default 0.3):

```
final_output = (1 - mem_bank_alpha) × word_scores + mem_bank_alpha × document_scores
```

This allows the model to leverage document-level repetition patterns — if the same word appeared earlier in the document and was correctly labeled, the memory bank provides strong prior evidence for the current prediction.

---

## 📝 WordRep — Word Representation

**File**: `wordrep.py` | **Class**: `WordRep(nn.Module)`

Constructs the word-level input representation by combining three complementary feature types.

### Components

| Component | Dimension | Source | Purpose |
|-----------|-----------|--------|---------|
| **Word Embeddings** | `word_emb_dim` (100) | Pre-trained (GloVe) or random | Semantic word meaning |
| **Char Features** | IntNet output dim | IntNet CNN | Morphological patterns (prefixes, suffixes, shape) |
| **Label-Similarity** | `num_labels` | Cosine similarity | Soft attention-like signal for label relevance |

### Label-Similarity (LS) Embeddings

A novel feature that provides each word with a distribution over label types:

```python
# For each word at each position:
LS_embs[batch, pos, label_id] = cosine_similarity(word_emb[batch, pos], label_emb[label_id])
```

This gives the model an explicit signal about which labels are most semantically similar to each word's embedding. For example, a word embedding close to the "PER" label embedding likely represents a person name.

### Output

```python
word_represent  # (batch, sent_len, word_emb_dim + char_features_dim)
LS_embs         # (batch, sent_len, num_labels) — label similarity scores
orig_word_embs  # (batch, sent_len, word_emb_dim) — raw word embeddings for memory bank
```

---

## 🔡 IntNet — Character-Level Inception CNN

**File**: `IntNet.py` | **Class**: `IntNet(nn.Module)`

An inception-style multi-scale convolutional neural network that extracts character-level features from words. Named after "Inception Network" for its parallel multi-scale convolution design.

### Architecture

```
Input: Character embeddings (batch, char_emb_dim, word_length)
  │
  ├──▶ Conv1d(kernel=3, pad=1) ──▶ ReLU ──┐
  │                                         │──▶ Concat ──▶ inception_features
  └──▶ Conv1d(kernel=5, pad=2) ──▶ ReLU ──┘
                                                │
         For each inception block (N blocks):   │
         ┌──────────────────────────────────────┘
         ▼
  Conv1d(kernel=1) ──▶ ReLU     ← bottleneck (reduce dimensions)
         │
         ├──▶ Conv1d(kernel=3) ──▶ ReLU ──┐
         │                                 │──▶ Concat ──┐
         └──▶ Conv1d(kernel=5) ──▶ ReLU ──┘              │──▶ Dense connection
                                                          │    (concat with ALL
         Previous inception_features ─────────────────────┘     previous features)
                                                │
                                                ▼
                                       GlobalMaxPool1d
                                                │
                                                ▼
                              (batch, char_hidden_dim) — one vector per word
```

### Key Design Choices

- **Multi-scale convolutions**: Kernel sizes 3 and 5 capture character trigrams and 5-grams in parallel
- **Dense connections**: Each block concatenates with ALL previous features (like DenseNet), preserving both low-level character patterns and high-level morphological features
- **1×1 bottleneck**: Reduces dimensionality before multi-scale convolutions to control parameter growth
- **Global max pooling**: Produces a fixed-size output regardless of word length

### Configuration

| Parameter | Config Key | Default | Effect |
|-----------|-----------|---------|--------|
| Embedding dim | `char_emb_dim` | 32 | Character embedding size |
| Hidden dim | `char_hidden_dim` | 16 | Intermediate convolution channels |
| Number of layers | `intNet_layer` | 7 | Total CNN layers → `(7-1)/2 = 3` inception blocks |
| Kernel types | `intNet_kernel_type` | 2 | Number of parallel kernels (2 = kernel 3 + kernel 5) |

### Output Dimension Calculation

```
output_dim = (intNet_layer - 1) / 2 × char_hidden_dim × kernel_type
           + char_emb_dim × 2 × kernel_type
         = (7 - 1) / 2 × 16 × 2 + 32 × 2 × 2
         = 3 × 32 + 128
         = 224
```

---

## 🌐 SentenceRep — Sentence-Level Feature Extractor

**File**: `SentenceRep.py` | **Class**: `SentenceRep(nn.Module)`

Extracts sentence-level global context for each word position using a **separate** feature extractor from the main BiLSTM.

### Why a Separate Extractor?

The sentence-level features need to capture the global sentence context **before** the main BiLSTM processes the data. Using a separate BiLSTM/CNN means the sentence representation is computed independently and then concatenated, providing complementary information.

### Important Implementation Detail

`SentenceRep` creates its **own** `WordRep` instance and calls it with `sentence_level=True`. This flag **disables** label-similarity computation in WordRep to avoid a circular dependency:

```
WordSequence uses label-similarity → which needs sentence features
SentenceRep provides sentence features → using its own WordRep (without LS)
```

### Feature Extractor Options

| Option | Config Value | Architecture |
|--------|-------------|--------------|
| **BiLSTM** (default) | `LSTM` | Bidirectional LSTM → dropout |
| **GRU** | `GRU` | Bidirectional GRU → dropout |
| **CNN** | `CNN` | Linear → tanh → stacked Conv1d + BatchNorm + ReLU |

### Output

```python
# (batch, sent_len, global_hidden_dim) — sentence context for each position
```

---

## 🗄️ MemoryBank — Document-Level Memory Network

**File**: `MemoryBank.py` | **Class**: `MemoryBank(nn.Module)`

Implements a document-level memory mechanism that stores and retrieves word representations across sentences within a document.

### How It Works

```
1. STORAGE (non-trainable Parameter matrices):
   ┌─────────────────────────────────────────────────────────────┐
   │  bankmem[total_word_instances, hidden_dim]  ← BiLSTM outputs│
   │  wordmem[total_word_instances, word_dim]    ← Word embeddings│
   └─────────────────────────────────────────────────────────────┘

2. RETRIEVAL (during forward pass):
   For each word in current batch:
   a. Find all co-occurring instances (same word type) via word_mat
   b. Compute cosine similarity between current word embedding and stored embeddings
   c. Apply masked softmax → attention weights
   d. Weighted sum of stored hidden representations → document context

3. UPDATE (after each training epoch):
   a. For correctly predicted words, store their current hidden states
   b. Track which words were correct via make_idx()
```

### Memory Bank Contents

| Matrix | Shape | Content | Updated When |
|--------|-------|---------|-------------|
| `bankmem` | (total_instances, hidden_dim) | BiLSTM output states | After each epoch |
| `wordmem` | (total_instances, word_dim) | Word embeddings | After each epoch |

### Why Non-Trainable?

The memory matrices are `nn.Parameter` with `requires_grad=False`. They serve as a **lookup table** updated by explicit assignment (not gradient descent). This avoids the prohibitive memory cost of backpropagating through the entire document history.

### Attention Mechanism

```python
# For a word at the current position:
score = cosine_similarity(current_word_emb, stored_word_embs)  # similarity to each stored instance
weights = masked_softmax(score)                                 # normalize over valid entries
doc_context = sum(weights * stored_hidden_states)               # weighted sum of hidden states
```

---

## 🔗 CRF — Conditional Random Field

**File**: `crf.py` | **Class**: `CRF(nn.Module)`

Implements a **linear-chain Conditional Random Field** for structured sequence prediction.

### Why CRF Instead of Softmax?

Softmax makes **independent** predictions at each position, which can produce invalid label sequences (e.g., `I-PER` without a preceding `B-PER`, or `B-PER` followed by `I-ORG`). The CRF models **dependencies between adjacent labels** through a learned transition matrix.

```
Softmax: P(y_t | x) — each position independent
CRF:     P(y_1, y_2, ..., y_T | x) — entire sequence jointly
```

### Core Components

#### 1. Transition Matrix

```python
self.transitions = nn.Parameter(torch.zeros(tagset_size + 2, tagset_size + 2))
# transitions[i][j] = score of transitioning from tag i to tag j
# +2 for START_TAG and STOP_TAG
```

This matrix learns constraints like:
- `B-PER → I-PER` gets a high score (valid continuation)
- `B-PER → I-ORG` gets a low score (entity type mismatch)
- `START_TAG → I-*` gets a low score (can't start with Inside)
- `O → I-*` gets a low score (Inside must follow Begin)

#### 2. Forward Algorithm (`_calculate_PZ`)

Computes the **partition function** Z — the sum of scores over ALL possible label sequences. Uses dynamic programming to avoid exponential enumeration:

```
Time complexity: O(T × K²)  where T = sequence length, K = number of tags
Space complexity: O(K)
```

This is equivalent to the "forward" step in a Hidden Markov Model, but in log-space for numerical stability.

#### 3. Viterbi Decoding (`_viterbi_decode`)

Finds the **highest-scoring** label sequence using dynamic programming with backpointers:

```
For each position t and each tag k:
    score[t][k] = max over all previous tags j of:
        score[t-1][j] + transition[j→k] + emission[t][k]
    backpointer[t][k] = argmax j of the above

Then trace back from the best final tag to recover the full sequence.
```

#### 4. Score Sentence (`_score_sentence`)

Computes the score of a **specific** (gold) label sequence by summing emission scores and transition scores:

```
score = Σ_t  emission[t][y_t] + transition[y_{t-1} → y_t]
      + transition[START → y_0] + transition[y_T → STOP]
```

#### 5. N-best Decoding (`_viterbi_decode_nbest`)

Extension of Viterbi that maintains the **top N** scoring sequences at each position, useful for confidence estimation and reranking.

### Loss Function

```
NLL = log(Z) - score(gold_sequence)
    = log(Σ over all sequences exp(score(y))) - score(y*)
```

Where `y*` is the gold (correct) label sequence.

---

## 📊 Data Flow

### Training Flow

```
Input sentence: "EU rejects German call"
  │
  ▼
┌─ WordRep ─────────────────────────────────────────────────────────────┐
│  Word embeddings:  [emb_EU, emb_rejects, emb_German, emb_call]       │
│  + IntNet chars:   [char_EU, char_rejects, char_German, char_call]    │
│  = word_represent: (4, word_dim + char_dim)                           │
│  + Label-Similarity: (4, num_labels)                                  │
└───────────────────────────────────────────────────────────────────────┘
  │                      │
  │                      ▼
  │  ┌─ SentenceRep + Label Attention ─────────────────────────────────┐
  │  │  Separate BiLSTM on word+char features                          │
  │  │  → Label attention using LS scores                              │
  │  │  = sentence_represent: (4, global_hidden_dim)                   │
  │  └─────────────────────────────────────────────────────────────────┘
  │                      │
  ▼                      ▼
┌─ Concatenate ─────────────────────────────────────────────────────────┐
│  combined = [word_represent || sentence_represent]                     │
│  shape: (4, word_dim + char_dim + global_hidden_dim)                  │
└───────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ BiLSTM ──────────────────────────────────────────────────────────────┐
│  Pack sequences → BiLSTM → Unpack → Dropout                          │
│  = feature_out: (4, hidden_dim)                                       │
└───────────────────────────────────────────────────────────────────────┘
  │
  ├──▶ hidden2tag → word_scores: (4, num_labels)
  │
  ├──▶ MemoryBank → doc_scores:  (4, num_labels)
  │
  ▼
┌─ Interpolation ──────────────────────────────────────────────────────┐
│  output = 0.7 × word_scores + 0.3 × doc_scores                       │
│  (emission scores for each label at each position)                    │
└───────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ CRF ────────────────────────────────────────────────────────────────┐
│  Loss = log(Z) - score([S-ORG, O, S-MISC, O])                        │
│  Decode: Viterbi → [S-ORG, O, S-MISC, O]                             │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Tensor Shapes Reference

| Tensor | Shape | Description |
|--------|-------|-------------|
| `word_inputs` | `(batch, sent_len)` | Word index tensor |
| `word_seq_lengths` | `(batch,)` | Actual sentence lengths (for packing) |
| `char_inputs` | `(batch × sent_len, max_word_len)` | Character indices (flattened) |
| `char_seq_lengths` | `(batch × sent_len,)` | Character lengths per word |
| `char_seq_recover` | `(batch × sent_len,)` | Recovery indices after sorting |
| `mask` | `(batch, sent_len)` | Binary mask: 1 = valid, 0 = padding |
| `batch_label` | `(batch, sent_len)` | Gold label indices |
| `batch_idx` | `(batch, sent_len)` | Global word instance indices |
| `word_embs` | `(batch, sent_len, word_emb_dim)` | Word embeddings |
| `char_features` | `(batch, sent_len, char_out_dim)` | IntNet character features |
| `LS_embs` | `(batch, sent_len, num_labels)` | Label-similarity scores |
| `sentence_represent` | `(batch, sent_len, global_hidden_dim)` | Sentence-level features |
| `feature_out` | `(batch, sent_len, hidden_dim)` | BiLSTM output |
| `outputs` | `(batch, sent_len, num_labels)` | Final emission scores |

---

## ⚙️ Hyperparameter Impact

| Hyperparameter | Config Key | Default | Affects | Impact |
|----------------|-----------|---------|---------|--------|
| BiLSTM hidden dim | `hidden_dim` | 256 | `WordSequence.lstm` | Larger = more capacity, slower training |
| Global hidden dim | `global_hidden_dim` | 128 | `SentenceRep.lstm` | Controls sentence-level feature richness |
| Char hidden dim | `char_hidden_dim` | 16 | `IntNet` conv layers | Controls character feature capacity |
| IntNet layers | `intNet_layer` | 7 | `IntNet` depth | More layers = deeper character patterns |
| IntNet kernels | `intNet_kernel_type` | 2 | `IntNet` parallel convolutions | 2 = kernel 3 + kernel 5 |
| Memory bank α | `mem_bank_alpha` | 0.3 | `WordSequence` interpolation | Higher = more document-level influence |
| Dropout | `dropout` | 0.5 | Embedding/feature dropout | Regularization strength |
| RNN dropout | `rnn_dropout` | 0.5 | BiLSTM output dropout | Regularization on sequence features |
| LSTM layers | `lstm_layer` | 1 | Number of stacked LSTM layers | More = deeper sequence modeling |
| Bidirectional | `bilstm` | True | LSTM direction | True = bidirectional (recommended) |
| Use CRF | `use_crf` | True | Decoding layer | True = structured decoding, higher F1 |
| Use char | `use_char` | True | Character features | True = include IntNet features |

---

## 🔬 Implementation Notes

### Why Two WordRep Instances?

The model contains two separate `WordRep` modules:
1. **Main WordRep** (in `WordSequence`) — computes word embeddings, char features, AND label-similarity
2. **Sentence WordRep** (in `SentenceRep`) — computes word embeddings and char features ONLY (no label-similarity)

This prevents a circular dependency: label-similarity is used to compute sentence features (via label-attention), so the sentence-level WordRep cannot also compute label-similarity.

### Char Input Handling

Character inputs are **pre-flattened** across the batch and sentence dimensions:
- Input shape: `(batch × sent_len, max_word_len)` — treats each word as an independent sample
- After IntNet: `(batch × sent_len, char_dim)` — one vector per word
- Reshaped back: `(batch, sent_len, char_dim)` — using `char_seq_recover` to restore original order

The flattening and recovery is necessary because words within a batch are **sorted by character length** for efficient padding, then un-sorted using `char_seq_recover`.

### Packed Sequences

Both the main BiLSTM (`WordSequence`) and the sentence BiLSTM (`SentenceRep`) use **packed sequences** via `pack_padded_sequence` / `pad_packed_sequence`. This ensures the LSTM doesn't waste computation on padding tokens and produces correct hidden states.
