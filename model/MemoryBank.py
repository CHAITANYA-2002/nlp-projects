"""
MemoryBank.py - Document-Level Memory Network

Implements a memory bank that stores and retrieves word representations
across sentences within a document, enabling document-level context for NER.

How it works:
1. The bank maintains two parameter matrices (non-trainable):
   - bankmem: Stores hidden representations (from BiLSTM output) for all words
   - wordmem: Stores word embeddings for all words

2. During forward pass:
   - For each word in the current batch, find its co-occurring words from
     a pre-built word matrix (word_mat) — words that share the same token
   - Compute attention weights via cosine similarity between the current
     word embedding and the stored embeddings of co-occurring words
   - Return a weighted sum of the stored hidden representations as the
     document-level context for each word

3. After each epoch, the bank is updated with the latest hidden states
   from correctly predicted words (via update() and make_idx()).

This mechanism allows the model to leverage document-level repetition patterns
(e.g., the same entity appearing multiple times in a document).
"""

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
import numpy as np 


class MemoryBank(nn.Module):
    """
    Document-level memory bank for storing and retrieving word representations.
    
    The bank stores hidden states and word embeddings for previously seen words,
    enabling document-level context via attention-based retrieval.
    
    Args:
        data: Data configuration object containing word matrix, embedding dims, etc.
    """
    def __init__(self, data):
        super(MemoryBank, self).__init__()
        self.gpu = data.HP_gpu

        total_num = data.word_idx         # Total number of word instances across the dataset
        self.word_dim = data.word_emb_dim
        self.word_mat = data.word_mat     # word_mat[word_id] -> list of instance indices with same word
        self.sent_dim = data.HP_hidden_dim

        # Memory bank for hidden representations (non-trainable)
        # Stores BiLSTM outputs for each word instance
        bankmem = torch.Tensor(total_num, self.sent_dim)
        nn.init.uniform_(bankmem, -1, 1)     
        self.bankmem = nn.Parameter(bankmem, requires_grad = False)
        self.bankmem.data[0] = torch.zeros(self.sent_dim)  # Index 0 is padding
        
        # Memory bank for word embeddings (non-trainable)
        # Stores word embeddings for computing retrieval attention
        wordmem = torch.Tensor(total_num, self.word_dim)
        nn.init.uniform_(wordmem, -1, 1)
        self.wordmem = nn.Parameter(wordmem, requires_grad = False) 
        self.wordmem.data[0] = torch.zeros(data.word_emb_dim)  # Index 0 is padding

        self.dropout = nn.Dropout(data.mem_bank_dropout)
        self.idx = None  # Tracks which words have been correctly predicted (for updates)
    

    def forward(self, word_idx, word_embs):
        """
        Retrieve document-level context from the memory bank.
        
        For each word, finds co-occurring word instances (same token type) and
        computes an attention-weighted sum of their stored hidden representations.
        
        Args:
            word_idx: (total_num,) - Global indices identifying each word instance.
            word_embs: (total_num, word_dim) - Current word embeddings.
            
        Returns:
            doc_hidden: (total_num, hidden_dim) - Document-level context vectors.
        """
        # If no words have been correctly predicted yet, return zeros
        if self.idx is None:
            return word_embs.new_full((word_embs.size(0), self.sent_dim), fill_value=0)
        
        num = word_embs.size(0)
        # For each word, get up to 500 co-occurring word instance indices
        idx = [list(self.word_mat[i][:500]) for i in word_idx]
        word_idx_len = list(map(len, idx))
        max_word_idx_len = max(word_idx_len)
        # Pad to uniform length
        idx = [idx[i] + [0] * (max_word_idx_len - len(idx[i])) for i in range(len(idx))]
        idx = torch.tensor(idx).type_as(self.idx).view(-1)

        # Create mask for valid (non-padded) co-occurring words
        mask = torch.zeros((num, max_word_idx_len), requires_grad=False).type_as(self.idx)
        for i in range(num):
            mask[i, :word_idx_len[i]] = torch.Tensor([1]*word_idx_len[i])

        # Cosine similarity attention: compare current word embedding with stored embeddings
        score = torch.bmm(F.normalize(word_embs.unsqueeze(1), dim=-1), F.normalize(self.wordmem[idx].view(word_embs.size(0), max_word_idx_len, word_embs.size(1)), dim=-1).transpose(2,1)).squeeze(1)
        score = self.partial_softmax(score, mask, 1)
                 
        # Weighted sum of stored hidden representations using attention scores
        doc_hidden = torch.bmm(score.unsqueeze(1), self.bankmem[idx].view(num, max_word_idx_len, -1)).squeeze(1)
        
        return doc_hidden
        
    def update(self, idx, word_embs, hidden_embs):
        """
        Update memory bank with latest representations from the current batch.
        
        Args:
            idx: (update_num,) - Global word instance indices to update.
            word_embs: (update_num, word_dim) - New word embeddings to store.
            hidden_embs: (update_num, hidden_dim) - New hidden states to store.
        """
        self.bankmem.data[idx] = hidden_embs.data 
        self.wordmem.data[idx] = word_embs.data

    def make_idx(self, idx_list):
        """
        Record which word instances were correctly predicted this epoch.
        Called after each training epoch to mark reliable entries in the bank.
        
        Args:
            idx_list: Tensor of global word instance indices that were correctly labeled.
        """
        if len(idx_list) == 0:
            self.idx = None
        else:
            self.idx = idx_list

    def partial_softmax(self, inputs, mask, dim):
        """
        Masked softmax that only considers valid (non-padded) positions.
        
        Args:
            inputs: (batch_size, seq_len) - Raw attention scores.
            mask: (batch_size, seq_len) - Binary mask (1=valid, 0=padded).
            dim: Dimension along which to normalize.
            
        Returns:
            Normalized attention weights with masked positions set to 0.
        """
        exp_inp = torch.exp(inputs)
        exp_inp_weighted = torch.mul(exp_inp, mask.float())
        exp_inp_sum = torch.sum(exp_inp_weighted, dim=dim, keepdim=True)
        partial_softmax_score = torch.div(exp_inp_weighted, exp_inp_sum)
        return partial_softmax_score