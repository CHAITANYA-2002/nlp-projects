"""
SentenceRep.py - Sentence-Level Global Feature Extractor

Extracts sentence-level contextual representations that capture global
context information for each word position. This provides the "sentence-level"
component of the hierarchical representation.

The module uses a separate WordRep instance (with sentence_level=True,
which disables label-similarity to avoid circular dependency) and passes
word+char features through either:
- BiLSTM (bidirectional LSTM)
- GRU (Gated Recurrent Unit)
- CNN (multi-layer 1D convolution with batch normalization)

The output is a global hidden representation (global_hidden_dim) for each
word position, which is later combined with the word-level representation
in WordSequence via label-attention.
"""

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep  # Word representation (word + char embeddings)
import numpy as np 


class SentenceRep(nn.Module):
    """
    Sentence-level feature extractor using BiLSTM, GRU, or CNN.
    
    Produces global contextual features (global_hidden_dim) for each word position,
    capturing sentence-level patterns that complement word-level features.
    
    Args:
        data: Data configuration object with model hyperparameters.
    """
    def __init__(self, data):
        super(SentenceRep, self).__init__()
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.droplstm = nn.Dropout(data.rnn_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer

        # Separate WordRep instance for sentence-level features
        # (sentence_level=True in forward to skip label-similarity computation)
        self.wordrep = WordRep(data)

        # Calculate input size: word_emb_dim + character feature dim
        self.input_size = data.word_emb_dim
        if self.use_char:
            kernel_type = data.HP_intNet_kernel_type
            char_dim = data.HP_char_hidden_dim
            # IntNet output dimension depends on layer count and kernel types
            self.input_size += int( (data.HP_intNet_layer - 1) // 2 * char_dim * kernel_type + char_dim * 2 * kernel_type)
       
        # BiLSTM splits hidden dim in half (each direction gets half)
        if self.bilstm_flag:
            lstm_hidden = data.global_hidden_dim // 2
        else:
            lstm_hidden = data.global_hidden_dim

        # Build the chosen feature extractor
        self.global_feature_extractor = data.global_feature_extractor
        if self.global_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.global_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.global_feature_extractor == "CNN":
            # CNN option: linear projection + stacked Conv1d with batch norm
            self.word2cnn = nn.Linear(self.input_size, data.global_hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = int((kernel-1)/2)
            for idx in range(self.cnn_layer):
                self.cnn_list.append(nn.Conv1d(data.global_hidden_dim, data.global_hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.rnn_dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.global_hidden_dim))
    
        # Move to GPU if available
        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            if self.global_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            else:
                self.lstm = self.lstm.cuda()


    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
        Extract sentence-level features for each word position.
        
        Args:
            word_inputs: (batch_size, sent_len) - Word indices.
            word_seq_lengths: (batch_size,) - Actual sequence lengths.
            char_inputs: (batch_size*sent_len, word_length) - Character indices.
            char_seq_lengths: (batch_size*sent_len,) - Character lengths.
            char_seq_recover: Recovery indices for character order.
            
        Returns:
            feature_out: (batch_size, sent_len, global_hidden_dim) - Sentence-level features.
        """
        # Get word+char representations (skip label-similarity with sentence_level=True)
        word_represent, _, _ = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, True)

        if self.global_feature_extractor == "CNN":
            # CNN path: project -> tanh -> stacked Conv1d with ReLU + dropout + batch norm
            batch_size = word_inputs.size(0)
            word_in = torch.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                if batch_size > 1:
                    cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2,1).contiguous()
        else:
            # LSTM/GRU path: pack -> process -> unpack -> dropout
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            lstm_out, hidden = self.lstm(packed_words, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            feature_out = self.droplstm(lstm_out.transpose(1,0))

        return feature_out
