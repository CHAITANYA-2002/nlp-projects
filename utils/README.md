# Utility Modules — Data Processing & Evaluation

This directory contains the core logic for data handling, vocabulary management, and sequence labeling evaluation metrics. These utilities serve as the bridge between raw text files and the neural network components in the `model/` directory.

---

## Table of Contents

- [File Reference](#file-reference)
- [Data — Configuration Hub](#data--configuration-hub)
- [Alphabet — Vocabulary Mapping](#alphabet--vocabulary-mapping)
- [Functions — Data I/O](#functions--data-io)
- [Metric — NER Evaluation](#metric--ner-evaluation)
- [TagSchemeConverter — Tag Conversions](#tagschemeconverter--tag-conversions)

---

## File Reference

| File | Primary Class/Function | Purpose |
|------|------------------------|---------|
| `__init__.py` | — | Python package marker |
| `data.py` | `Data` | Central configuration hub, alphabet management, and instance generation |
| `alphabet.py` | `Alphabet` | Bidirectional mapping between tokens (words/chars/labels) and their indices |
| `functions.py` | `read_instance()` | Data I/O: reading CoNLL files and loading pretrained embeddings |
| `metric.py` | `get_ner_fmeasure()` | Evaluation metrics (P, R, F1) for BIO/BIOES entity spans |
| `tagSchemeConverter.py` | `IOB2BIO`, `BIO2BIOES` | Converts between different NER tag representation schemes |

---

## Data — Configuration Hub

**File**: `data.py` | **Class**: `Data`

The `Data` class is the "brain" of the preprocessing pipeline. It stores every setting, alphabet, and data instance used throughout the project.

### Key Responsibilities

1. **Config Management**: Parses `.config` files and maps them to class attributes (e.g., `self.HP_lr`, `self.use_crf`).
2. **Alphabet Building**: Orchestrates the scanning of datasets to build vocabularies for words, characters, and labels.
3. **Embedding Loading**: Manages the loading and normalization of pretrained GloVe/BERT/Label embeddings.
4. **Instance Generation**: Converts raw text strings into numerical ID tensors (`torch.LongTensor`) ready for the `SeqLabel` model.

### Data Split Storage

| Attribute | Description |
|-----------|-------------|
| `self.train_Ids` | Numerical instances of the training set |
| `self.dev_Ids` | Numerical instances of the validation set |
| `self.test_Ids` | Numerical instances of the test set |

---

## Alphabet — Vocabulary Mapping

**File**: `alphabet.py` | **Class**: `Alphabet`

Manages a bidirectional mapping between strings and integer IDs.

- **String → ID**: Used during preprocessing to convert text to indices.
- **ID → String**: Used during decoding to convert model predictions back to readable tags.

### Special Handling

- **Padding**: Index `0` is reserved and holds no token of its own. `size()`
  therefore returns one more than the number of stored entries.
- **Seeded entries**: A non-label alphabet is seeded with `<PAD>` at index `1`
  and `</unk>` at index `2`, so user tokens start at `3`. A label alphabet
  (`label=True`) is not seeded at all, and its first label takes index `1`.
- **Freezing**: `close()` stops the alphabet from adding new entries, keeping
  indices consistent between training and inference. A closed word alphabet maps
  unseen tokens to `</unk>`; a closed **label** alphabet has no such fallback and
  raises `KeyError` on an unseen label.

`tests/test_alphabet.py` pins all of the above.

---

## Functions — Data I/O

**File**: `functions.py`

Low-level helper functions for reading files and loading embeddings.

- **`read_instance()`**: The primary data parser. It handles the column format
  (tokens, chars, and labels) and splits sentences on blank lines. Each instance
  is `[words, features, chars, labels, idxs]`, where `idxs` is the corpus-wide
  word position the memory bank keys on. With `number_normalized=True` the
  recorded `words` keep their original surface form while the ids — and the
  characters handed to IntNet — use the digit-collapsed form.
- **`build_pretrain_embedding()`**: Builds one embedding row per vocabulary entry.
  Lookup is exact-case first, then lowercase (GloVe is lowercased, the vocabulary
  is not), and anything still unmatched gets a uniform random row scaled by
  `sqrt(3/dim)`.
- **`normalize_word()`**: Replaces every digit with `0`, so `1996` and `2024`
  collapse to the same vocabulary entry.
- **`norm2one()`**: Normalizes a vector to unit length (L2 norm).

`tests/test_functions.py` pins the parser and both embedding paths.

---

## Metric — NER Evaluation

**File**: `metric.py` | **Function**: `get_ner_fmeasure()`

Implements standard NER evaluation metrics (Accuracy, Precision, Recall, and F1-score) based on **entity span matching**, not just token-level accuracy.

### Features

- Supports both **BIO** and **BIOES** tag schemes (`BMES` is accepted as a synonym
  for `BIOES`).
- Correctly identifies multi-token entity boundaries (e.g. `B-PER I-PER E-PER`).
- Scores are **micro-averaged**: every predicted and gold span from every sentence
  is pooled into one set, and precision and recall are computed over those totals.
  There is no per-type breakdown, so a frequent entity type dominates the score.
- Spans are compared as strings of the form `[start,end]TYPE`, matched exactly.
  A predicted span with the right type but the wrong boundary counts as both a
  false positive and a false negative.
- When nothing is predicted, precision is returned as the sentinel `-1` rather
  than raising, and F1 becomes `-1` too. `main.py` prints those values verbatim,
  so a reported `f: -1.0000` means "no entities predicted", not "F1 of -1".

`tests/test_metric.py` pins the span extraction and each of these edge cases.

---

## TagSchemeConverter — Tag Conversions

**File**: `tagSchemeConverter.py`

A standalone utility for preparing datasets in the correct format.

| Conversion | Usage |
|------------|-------|
| **IOB → BIO** | Fixes invalid I- tags that don't follow a B- tag. |
| **BIO → BIOES** | Upsamples BIO tags to BIOES (adds End and Single tags) for better boundary detection. |
| **BIOES → BIO** | Standardizes predictions for external tools that only support 3-tag BIO. |

**Example usage**:
```bash
python utils/tagSchemeConverter.py BIO2BIOES train.bio train.bioes
```
