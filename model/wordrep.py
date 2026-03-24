"""
wordrep.py - Word Representation Module

Constructs the word-level input representation by combining:
1. Pre-trained word embeddings (e.g., GloVe)
2. Character-level features from IntNet (inception-style CNN)
3. Label-Similarity (LS) embeddings: cosine similarity between each word
   embedding and all label embeddings, providing label-aware context

The label-similarity mechanism computes a soft attention-like distribution
over label types for each word position, giving the model explicit signal
about which labels are most similar to each word's semantic representation.

Output:
    - word_represent: Concatenation of [word_emb, char_features]
    - LS_embs: Label-similarity scores (batch_size, sent_len, num_labels)
    - orig_word_embs: Original word embeddings (for memory bank)
"""

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
from .IntNet import IntNet  # Inception-style character CNN
import torch.nn.functional as F   
import numpy as np 

# Set random seeds for reproducibility
seed_num=42
torch.manual_seed(seed_num)
np.random.seed(seed_num)
torch.cuda.manual_seed(seed_num)


class WordRep(nn.Module):
    """
    Word Representation layer combining word embeddings, character features,
    and label-similarity embeddings.
    
    Args:
        data: Data configuration object with embedding dimensions, alphabet sizes,
              and pretrained embedding matrices.
    """
    def __init__(self, data):
        super(WordRep, self).__init__()
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.batch_size = data.HP_batch_size
        
        # Character-level feature extractor (IntNet: multi-scale inception CNN)
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            self.char_feature = IntNet(data.char_alphabet.size(), self.char_embedding_dim, data.HP_intNet_layer, data.HP_intNet_kernel_type, data.HP_dropout, self.char_hidden_dim, self.gpu)
           
        # Word embedding layer - can be initialized with pretrained embeddings (e.g., GloVe)
        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        # Label embedding layer - embeds each NER label type into the same space as words
        # Used for computing cosine similarity between word and label representations
        self.label_embedding_dim = data.word_emb_dim
        self.label_alphabet_size = data.label_alphabet.size()
        self.label_embedding = nn.Embedding(self.label_alphabet_size, self.label_embedding_dim)
        # Create a tensor of all label indices [0, 1, 2, ..., num_labels-1]
        self.label_type = torch.from_numpy(np.array([i for i in range(self.label_alphabet_size)]))
        if data.pretrain_label_embedding is not None:
            self.label_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_label_embedding))
        else:
            self.label_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(self.label_alphabet_size, self.label_embedding_dim)))

        # Cosine similarity module for label-word similarity computation
        self.cos_embs = nn.CosineSimilarity(dim=-1)

        # Move modules to GPU if available
        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()
            self.label_embedding = self.label_embedding.cuda()
            self.cos_embs = self.cos_embs.cuda()
            self.label_type = self.label_type.cuda()
         

    def random_embedding(self, vocab_size, embedding_dim):
        """
        Generate random embeddings with uniform distribution scaled by sqrt(3/dim).
        Used as fallback when no pretrained embeddings are available.
        
        Args:
            vocab_size: Number of entries in the embedding table.
            embedding_dim: Dimensionality of each embedding vector.
            
        Returns:
            numpy array of shape (vocab_size, embedding_dim)
        """
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

   
    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sentence_level=False):
        """
        Build word representations combining word embeddings, char features, and label similarity.
        
        Args:
            word_inputs: (batch_size, sent_len) - Word index tensor.
            word_seq_lengths: (batch_size,) - Actual sequence lengths.
            char_inputs: (batch_size*sent_len, word_length) - Character indices.
            char_seq_lengths: (batch_size*sent_len,) - Character sequence lengths.
            char_seq_recover: Recovery indices to restore character order after sorting.
            sentence_level: If True, skip label similarity (used for SentenceRep).
            
        Returns:
            word_represent: (batch_size, sent_len, hidden_dim) - Combined word+char features.
            LS_embs: (batch_size, sent_len, num_labels) - Label similarity scores, or None.
            orig_word_embs: (batch_size, sent_len, word_dim) - Original word embeddings.
        """
        batch_size, sent_len = word_inputs.size()[:2]
        word_embs =  self.word_embedding(word_inputs)
        word_list = [word_embs]
        orig_word_embs = word_embs
      
        # Extract character-level features using IntNet and append to word representations
        if self.use_char:
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]  # Restore original word order
            char_features = char_features.view(batch_size,sent_len,-1)
            word_list.append(char_features) 

        # Compute label-similarity embeddings (cosine similarity between words and labels)
        # Skip this for sentence-level representation to avoid circular dependency
        if not sentence_level:          
            label_embs = self.label_embedding(self.label_type)  # (num_labels, label_dim)
            # Expand word embeddings to compare with each label: (batch, sent_len, num_labels, dim)
            emb_batch = orig_word_embs.unsqueeze(2).repeat(1, 1, self.label_alphabet_size, 1) 
            # Expand label embeddings to match batch/sentence dimensions
            new_label_emb = label_embs.unsqueeze(0).unsqueeze(0).repeat(batch_size, sent_len, 1, 1) 
            # Cosine similarity between each word and all labels -> (batch, sent_len, num_labels)
            LS_embs = self.drop(self.cos_embs(emb_batch, new_label_emb).view(batch_size, sent_len, -1))
        else:
            LS_embs = None
            
        # Concatenate word embeddings with character features
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent, LS_embs, orig_word_embs
