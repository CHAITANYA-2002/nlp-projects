"""
IntNet.py - Inception-style Multi-Scale Character CNN (IntNet)

Implements an inception-style convolutional neural network for extracting
character-level features from words. Instead of using a single kernel size,
IntNet uses multiple parallel convolution streams (kernel sizes 3 and 5)
to capture character n-grams at different scales.

Architecture:
    1. Character embedding layer
    2. Initial parallel convolution: Conv1d(kernel=3) and Conv1d(kernel=5)
    3. Stacked inception blocks (configurable depth):
       - 1x1 bottleneck convolution (reduces channel dimension)
       - Parallel Conv1d(kernel=3) + Conv1d(kernel=5)
       - Dense connection: concatenate with all previous features
    4. Max pooling over the sequence dimension to get a fixed-size output

Dense connections between layers allow the network to capture both
low-level character patterns and high-level morphological features.

The number of inception blocks is controlled by (cnn_layer - 1) / 2.
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IntNet(nn.Module):
    """
    Inception-style character-level CNN with dense connections.
    
    Args:
        alphabet_size: Number of unique characters in the alphabet.
        embedding_dim: Character embedding dimension.
        cnn_layer: Total number of CNN layers (controls inception block count).
        kernel_type: Number of kernel sizes (typically 2: kernel 3 and kernel 5).
        dropout: Dropout rate applied to character embeddings.
        hidden_size: Hidden dimension for intermediate convolution layers.
        gpu: Whether to use GPU acceleration.
    """
    def __init__(self, alphabet_size, embedding_dim, cnn_layer, kernel_type, dropout, hidden_size, gpu):
        super(IntNet, self).__init__()
        self.gpu = gpu
        self.cnn_layer = cnn_layer
        self.kernel_type = kernel_type  # Number of parallel kernel sizes (2 = use both 3 and 5)
        self.char_drop = nn.Dropout(dropout)

        # Character embedding layer - initialized with random embeddings
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(alphabet_size, embedding_dim)))

        # Initial parallel convolutions with different kernel sizes
        self.init_char_cnn_3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)  # Captures trigrams
        self.init_char_cnn_5 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2)  # Captures 5-grams
       
        # Stacked inception blocks with dense connections
        self.cnn_list = nn.ModuleList()         # 1x1 bottleneck convolutions
        self.multi_cnn_list_3 = nn.ModuleList() # Kernel=3 convolutions in each block
        self.multi_cnn_list_5 = nn.ModuleList() # Kernel=5 convolutions in each block

        # Track input dimension growth from dense connections
        last_dim = embedding_dim * self.kernel_type  # After initial parallel conv: 2 * embedding_dim
        for idx in range(int((self.cnn_layer - 1) / 2)):
            # 1x1 bottleneck: reduce accumulated features to hidden_size
            self.cnn_list.append(nn.Conv1d(last_dim, hidden_size, kernel_size=1, padding=0))
            # Parallel multi-scale convolutions
            self.multi_cnn_list_3.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1))
            self.multi_cnn_list_5.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2))
            # Dense connection: new features are concatenated with all previous features
            last_dim += hidden_size * self.kernel_type 

        # Move all layers to GPU if available
        if self.gpu:
            self.char_drop = self.char_drop.cuda()
            self.char_embeddings = self.char_embeddings.cuda()
            self.init_char_cnn_3 = self.init_char_cnn_3.cuda()
            self.init_char_cnn_5 = self.init_char_cnn_5.cuda()
            for idx in range(int((self.cnn_layer - 1) / 2)):
                self.cnn_list[idx] = self.cnn_list[idx].cuda()
                self.multi_cnn_list_3[idx] = self.multi_cnn_list_3[idx].cuda()
                self.multi_cnn_list_5[idx] = self.multi_cnn_list_5[idx].cuda()

                
    def random_embedding(self, vocab_size, embedding_dim):
        """Generate random embeddings scaled by sqrt(3/dim) for uniform initialization."""
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_last_hiddens(self, input, seq_lengths):
        """
        Extract character-level features using the full inception network.
        Applies max-pooling over the character sequence to produce a fixed-size
        representation for each word.
        
        Args:
            input: (batch_size, word_length) - Character index tensor.
            seq_lengths: (batch_size,) - Actual character sequence lengths (numpy array).
            
        Returns:
            (batch_size, char_hidden_dim) - Fixed-size character feature for each word.
        """
        batch_size, max_seq = input.size()

        activate_func = F.relu  # ReLU activation for all convolution layers

        # Embed characters and transpose for Conv1d: (batch, channels, seq_len)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2,1).contiguous()

        # Initial parallel convolutions: kernel=3 (trigrams) and kernel=5 (5-grams)
        char_cnn_out3 = activate_func(self.init_char_cnn_3(char_embeds))
        char_cnn_out5 = activate_func(self.init_char_cnn_5(char_embeds))

        # Concatenate initial multi-scale features
        last_cnn_feature = torch.cat([char_cnn_out3,  char_cnn_out5], 1)  

        # Stack inception blocks with dense connections
        for idx in range(int((self.cnn_layer - 1) / 2)):
            # 1x1 bottleneck convolution to reduce dimensionality
            cnn_feature = activate_func(self.cnn_list[idx](last_cnn_feature)) 
            # Parallel multi-scale convolutions on the bottleneck output
            cnn_feature_3 = activate_func(self.multi_cnn_list_3[idx](cnn_feature))
            cnn_feature_5 = activate_func(self.multi_cnn_list_5[idx](cnn_feature)) 

            # Concatenate kernel=3 and kernel=5 outputs
            cnn_feature = torch.cat([cnn_feature_3,  cnn_feature_5], 1) 
            # Dense connection: concatenate with ALL previous features
            cnn_feature = torch.cat([cnn_feature, last_cnn_feature], 1)
            last_cnn_feature = cnn_feature

        # Global max pooling over the character sequence dimension
        char_cnn_out = last_cnn_feature
        char_cnn_out_max = F.max_pool1d(char_cnn_out, char_cnn_out.size(2))

        return char_cnn_out_max.view(batch_size, -1)

    def get_all_hiddens(self, input, seq_lengths):
        """
        Get character features for all positions (without max-pooling).
        
        Args:
            input: (batch_size, word_length) - Character indices.
            seq_lengths: (batch_size,) - Character sequence lengths.
            
        Returns:
            (batch_size, word_length, char_hidden_dim) - Per-position features.
        """
        batch_size = input.size(0)
        char_embeds = self.char_drop(self.char_embeddings(input))
        char_embeds = char_embeds.transpose(2,1).contiguous()
        char_cnn_out = self.char_cnn(char_embeds).transpose(2,1).contiguous()
        return char_cnn_out

    def forward(self, input, seq_lengths):
        """Default forward pass returns all hidden states (per-position features)."""
        return self.get_all_hiddens(input, seq_lengths)
