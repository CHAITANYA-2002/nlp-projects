# 📂 Sample Data — CoNLL Format NER Data

This directory contains sample training data for the Hierarchical Contextualized NER system.

---

## 📄 Data Format: CoNLL Column Format

The model expects data in a space-separated column format, also known as the **CoNLL 2003 format**. 

### Rules:
1. **One token per line**: Each line represents exactly one word/token.
2. **Tab or space separated**: The first column is the token, and the last column is the NER tag.
3. **Empty lines**: A blank line separates individual sentences.

### Example (BIOES Scheme):
```
EU      S-ORG
reject  O
German  S-MISC
call    O
.       O

Peter      B-PER
Blackburn  E-PER
```

---

## 🏷️ Supported Tag Schemes

The model primarily supports **BIOES** (Begin, Inside, Outside, End, Single) for optimal performance, but can also handle **BIO**.

| Tag | Full Name | Meaning |
|-----|-----------|---------|
| **B-** | Begin | First token of a multi-token entity |
| **I-** | Inside | Intermediate token of a multi-token entity |
| **E-** | End | Final token of a multi-token entity |
| **S-** | Single | A single-token entity |
| **O** | Outside | Not part of any named entity |

> **Pro Tip**: Use the `utils/tagSchemeConverter.py` script to convert your data from IOB or BIO to BIOES before training.

---

## 📁 Files in this Directory

- `eng.bioes.train`: A sample training file containing a few sentences formatted with BIOES tags.

Use these files to verify your installation and ensure your custom data follows the same structural pattern.
