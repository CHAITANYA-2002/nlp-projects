# 🧠 Hierarchical Contextualized Representation for Named Entity Recognition



This model introduces a **hierarchical contextualized representation** framework that captures character-level, word-level, sentence-level, and document-level features for Named Entity Recognition (NER). It achieves state-of-the-art results on standard NER benchmarks by integrating:

1. **IntNet** — An inception-style multi-scale character-level CNN
2. **Label-attention sentence representation** — Cosine similarity between word embeddings and label embeddings for global sentence context
3. **Memory Bank** — A document-level memory network that stores and retrieves representations of previously seen words
4. **BiLSTM-CRF** — Bidirectional LSTM with Conditional Random Field for sequence labeling

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SeqLabel (Top-level Model)                  │
│        ┌─────────────────────────┐   ┌──────────────────┐          │
│        │     WordSequence        │   │       CRF        │          │
│        │  ┌──────────────────┐   │   │  (Viterbi decode │          │
│        │  │    WordRep       │   │   │   + NLL loss)    │          │
│        │  │  ┌─────────┐    │   │   └──────────────────┘          │
│        │  │  │ IntNet  │    │   │                                  │
│        │  │  │(CharCNN)│    │   │                                  │
│        │  │  └─────────┘    │   │                                  │
│        │  │  Word Embeddings│   │                                  │
│        │  │  Label Embeddings│  │                                  │
│        │  └──────────────────┘   │                                  │
│        │  ┌──────────────────┐   │                                  │
│        │  │  SentenceRep     │   │                                  │
│        │  │(Global BiLSTM/CNN)│  │                                  │
│        │  └──────────────────┘   │                                  │
│        │  ┌──────────────────┐   │                                  │
│        │  │   MemoryBank     │   │                                  │
│        │  │ (Document-level) │   │                                  │
│        │  └──────────────────┘   │                                  │
│        │  BiLSTM Feature Extract │                                  │
│        └─────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
NLP/
├── main.py                     # Entry point: training, evaluation, and testing pipeline
├── demo.train.config           # Training configuration (I/O, network, hyperparameters)
├── demo.test.config            # Testing configuration (loads saved model & dataset)
├── model/                      # Neural network model components
│   ├── __init__.py             # Package marker
│   ├── seqlabel.py             # SeqLabel: top-level sequence labeling model (loss + decode)
│   ├── wordsequence.py         # WordSequence: main feature extractor combining all levels
│   ├── wordrep.py              # WordRep: word + char + label embedding representations
│   ├── IntNet.py               # IntNet: inception-style multi-scale character CNN
│   ├── SentenceRep.py          # SentenceRep: sentence-level global feature extractor
│   ├── MemoryBank.py           # MemoryBank: document-level memory network
│   └── crf.py                  # CRF: Conditional Random Field (forward, Viterbi, n-best)
├── utils/                      # Utility modules for data processing and evaluation
│   ├── __init__.py             # Package marker
│   ├── data.py                 # Data: config parsing, alphabet building, instance generation
│   ├── alphabet.py             # Alphabet: bidirectional mapping between tokens and indices
│   ├── functions.py            # Helper functions: data reading, pretrained embeddings
│   ├── metric.py               # NER evaluation metrics (precision, recall, F1-score)
│   └── tagSchemeConverter.py   # Tag scheme converter: IOB ↔ BIO ↔ BIOES
├── BERT/                       # BERT embedding generation tools
│   ├── __init__.py             # Google BERT license header
│   ├── get_aligned_bert_emb.py # Aligns BERT subword embeddings to token-level
│   ├── extract_features.py     # Google BERT feature extraction (unmodified)
│   ├── modeling.py             # Google BERT model architecture (unmodified)
│   ├── tokenization.py         # Google BERT tokenizer (unmodified)
│   └── run.sh                  # Shell script for running BERT extraction pipeline
├── sample_data/                # Sample training data
│   └── eng.bioes.train         # Example CoNLL format NER data (BIOES tag scheme)
├── LICENSE                     # Apache 2.0 License
└── README.md                   # This file
```

---

## 🛠️ Requirements

- **Python** 3.6 or higher
- **PyTorch** 0.4.1 or higher
- **NumPy**

---

## 🚀 Getting Started

### 1. Prepare Data

Place your NER data in `sample_data/` using **CoNLL format** (space-separated, one token per line, empty line between sentences):

```
EU      S-ORG
reject  O
German  S-MISC
call    O
.       O

Peter      B-PER
Blackburn  E-PER
```

Supported tag schemes: **BIOES** (recommended) and **BIO**.

### 2. Prepare Embeddings

- **Word embeddings**: GloVe or similar (e.g., `sample_data/eng.glove`)
- **Label embeddings**: Pre-trained label embeddings (e.g., `sample_data/eng.label`)
- **BERT embeddings** (optional): Use the `BERT/` tools to generate aligned embeddings

### 3. Configure

Edit `demo.train.config` to set:
- **I/O paths**: training/dev/test data, model save directory, embedding files
- **Network settings**: CRF, character features (IntNet), word feature extractor (LSTM)
- **Hyperparameters**: learning rate, dropout, hidden dimensions, etc.

### 4. Train

```bash
python main.py --config demo.train.config
```

The model will:
- Build word/char/label alphabets from training data
- Load pretrained embeddings
- Train for the configured number of epochs
- Evaluate on dev/test sets after each epoch
- Save the best model based on dev F1-score

### 5. Test

```bash
python main.py --config demo.test.config
```

---

## ⚙️ Configuration Reference

### I/O Settings

| Parameter       | Description                           | Example                          |
|----------------|---------------------------------------|----------------------------------|
| `train_dir`    | Training data file path               | `sample_data/eng.bioes.train`    |
| `dev_dir`      | Development data file path            | `sample_data/eng.bioes.dev`      |
| `test_dir`     | Test data file path                   | `sample_data/eng.bioes.test`     |
| `model_dir`    | Model save directory prefix           | `result/lstmcrf`                 |
| `word_emb_dir` | Pretrained word embeddings file       | `sample_data/eng.glove`          |
| `label_emb_dir`| Pretrained label embeddings file      | `sample_data/eng.label`          |

### Network Configuration

| Parameter                  | Description                              | Default  |
|---------------------------|------------------------------------------|----------|
| `use_crf`                 | Use CRF layer for sequence decoding      | `True`   |
| `use_char`                | Use character-level features             | `True`   |
| `word_seq_feature`        | Word-level feature extractor             | `LSTM`   |
| `global_feature_extractor`| Sentence-level feature extractor         | `LSTM`   |
| `char_seq_feature`        | Character-level feature extractor        | `IntNet` |

### Hyperparameters

| Parameter          | Description                            | Default  |
|-------------------|----------------------------------------|----------|
| `hidden_dim`      | BiLSTM hidden dimension                | `256`    |
| `global_hidden_dim`| Sentence-level hidden dimension       | `128`    |
| `char_hidden_dim` | Character feature hidden dimension     | `16`     |
| `intNet_layer`    | Number of IntNet CNN layers            | `7`      |
| `intNet_kernel_type`| IntNet kernel types (3 and 5)        | `2`      |
| `dropout`         | General dropout rate                   | `0.5`    |
| `rnn_dropout`     | RNN output dropout rate                | `0.5`    |
| `mem_bank_alpha`  | Memory bank interpolation weight       | `0.3`    |
| `learning_rate`   | Initial learning rate                  | `0.015`  |
| `lr_decay`        | Learning rate decay factor             | `0.05`   |
| `batch_size`      | Training batch size                    | `10`     |
| `iteration`       | Number of training epochs              | `70`     |
| `optimizer`       | Optimization algorithm                 | `SGD`    |

---

## 🔧 BERT Embeddings (Optional)

To generate BERT embeddings aligned to token boundaries:

```bash
cd BERT/
bash run.sh
```

This runs Google's BERT `extract_features.py` and then `get_aligned_bert_emb.py` to produce token-level embeddings from BERT's subword tokenization. Three alignment modes are supported:

| Mode    | Description                                          |
|---------|------------------------------------------------------|
| `first` | Use the embedding of the first subword piece         |
| `mean`  | Average all subword piece embeddings                 |
| `max`   | Element-wise maximum across subword piece embeddings |

---

## 📊 Evaluation

The model evaluates using standard NER metrics:
- **Accuracy** — Token-level prediction accuracy
- **Precision** — Fraction of predicted entities that are correct
- **Recall** — Fraction of gold entities that are predicted
- **F1-Score** — Harmonic mean of precision and recall

Evaluation supports both **BIO** and **BIOES** tag schemes with proper entity span matching.

---

## 🏗️ Tag Scheme Conversion

The `utils/tagSchemeConverter.py` utility converts between NER tag schemes:

```bash
# IOB to BIO
python utils/tagSchemeConverter.py IOB2BIO input_file output_file

# BIO to BIOES
python utils/tagSchemeConverter.py BIO2BIOES input_file output_file

# BIOES to BIO
python utils/tagSchemeConverter.py BIOES2BIO input_file output_file

# IOB to BIOES (two-step)
python utils/tagSchemeConverter.py IOB2BIOES input_file output_file
```

---

## 🔗 Pre-trained Models

A pre-trained model is available: [lstmcrf.model](https://drive.google.com/drive/folders/1G3kN1WsPJDVk9FdVUtIdv7DXd55p3yv0?usp=sharing)

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{luo2020hierarchical,
  title={Hierarchical Contextualized Representation for Named Entity Recognition},
  author={Luo, Ying and Xiao, Fengshun and Zhao, Hai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [NCRF++](https://github.com/jiesutd/NCRFpp) — The base framework for neural sequence labeling
- [Google BERT](https://github.com/google-research/bert) — Pre-trained language model for contextual embeddings
- [GloVe](https://nlp.stanford.edu/projects/glove/) — Pre-trained word vectors
