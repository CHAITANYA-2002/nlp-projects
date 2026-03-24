# 🛠️ Utility Modules — Data Processing & Evaluation

This directory contains the core logic for data handling, vocabulary management, and sequence labeling evaluation metrics. These utilities serve as the bridge between raw text files and the neural network components in the `model/` directory.

---

## Table of Contents

- [File Reference](#-file-reference)
- [Data — Configuration Hub](#-data--configuration-hub)
- [Alphabet — Vocabulary Mapping](#-alphabet--vocabulary-mapping)
- [Functions — Data I/O](#-functions--data-io)
- [Metric — NER Evaluation](#-metric--ner-evaluation)
- [TagSchemeConverter — Tag Conversions](#-tagschemeconverter--tag-conversions)

---

## 📁 File Reference

| File | Primary Class/Function | Purpose |
|------|------------------------|---------|
| `__init__.py` | — | Python package marker |
| `data.py` | `Data` | Central configuration hub, alphabet management, and instance generation |
| `alphabet.py` | `Alphabet` | Bidirectional mapping between tokens (words/chars/labels) and their indices |
| `functions.py` | `read_instance()` | Data I/O: reading CoNLL files and loading pretrained embeddings |
| `metric.py` | `get_ner_fmeasure()` | Evaluation metrics (P, R, F1) for BIO/BIOES entity spans |
| `tagSchemeConverter.py` | `IOB2BIO`, `BIO2BIOES` | Converts between different NER tag representation schemes |

---

## ⚙️ Data — Configuration Hub

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

## 🔤 Alphabet — Vocabulary Mapping

**File**: `alphabet.py` | **Class**: `Alphabet`

Manages a bidirectional mapping between strings and integer IDs.

- **String → ID**: Used during preprocessing to convert text to indices.
- **ID → String**: Used during decoding to convert model predictions back to readable tags.

### Special Handling

- **Padding**: Automatically reserves index `0` for the `<pad>` token.
- **Unknowns**: Automatically reserves index `1` for the `<unk>` token.
- **Freezing**: The `close()` method stops the alphabet from adding new entries, ensuring consistent indexing during validation and testing.

---

## 📥 Functions — Data I/O

**File**: `functions.py`

Low-level helper functions for reading files and loading embeddings.

- **`read_instance()`**: The primary data parser. It handles the specific column format (tokens, chars, and labels) and correctly identifies sentence boundaries (blank lines).
- **`build_pretrain_embedding()`**: Efficiently loads large embedding files (like GloVe), only keeping vectors for words that exist in the project's current vocabulary to save memory.
- **`norm2one()`**: Normalizes a vector to unit length (L2 norm).

---

## 📊 Metric — NER Evaluation

**File**: `metric.py` | **Function**: `get_ner_fmeasure()`

Implements standard NER evaluation metrics (Accuracy, Precision, Recall, and F1-score) based on **entity span matching**, not just token-level accuracy.

### Features

- Supports both **BIO** and **BIOES** tag schemes.
- Correctly identifies multi-token entity boundaries (e.g., `B-PER I-PER E-PER`).
- Calculates statistics for each entity type (PER, ORG, LOC, etc.) individually, then macro-averages them.

---

## 🏗️ TagSchemeConverter — Tag Conversions

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
