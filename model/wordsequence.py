"""
wordsequence.py - Hierarchical Word Sequence Feature Extractor

The central component that integrates all levels of the hierarchical
contextualized representation:

1. WORD-LEVEL: Word + character embeddings via WordRep
   - Pre-trained word embeddings (e.g., GloVe)
   - IntNet character features (multi-scale inception CNN)
   - Label-similarity (LS) embeddings

2. SENTENCE-LEVEL: Global sentence context via SentenceRep + label attention
   - SentenceRep extracts global features using a separate BiLSTM/GRU/CNN
   - Label-attention: uses LS scores as attention weights over sentence features
   - Produces a sentence representation that is concatenated with word features

3. WORD SEQUENCE: BiLSTM processes the combined (word + sentence) features
   - Packed sequence handling for variable-length sentences
   - Projects BiLSTM output to label space via hidden2tag linear layer

4. DOCUMENT-LEVEL: Memory bank for cross-sentence context via MemoryBank
   - Stores hidden states of previously seen words across the document
   - Retrieves context via cosine attention over co-occurring words
   - Interpolated with word-level predictions: output = (1-α)*word + α*doc

The final output is a tensor of emission scores for each label at each position,
which is consumed by either CRF or softmax for sequence decoding.
"""

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep       # Word + char + label embeddings
import numpy as np 
from .MemoryBank import MemoryBank  # Document-level memory network
from .SentenceRep import SentenceRep  # Sentence-level global feature extractor


class WordSequence(nn.Module):
    """
    Hierarchical feature extractor combining word, sentence, and document context.
    
    This is the core module that produces emission scores for each label at each
    word position, integrating features from all three hierarchical levels.
    
    Args:
        data: Data configuration object with all hyperparameters and settings.
    """
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.data = data
        self.use_char = data.use_char
        self.label_alphabet_size = data.label_alphabet.size()
        self.droplstm = nn.Dropout(data.rnn_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer

        # Word representation module (word emb + char features + label similarity)
        self.wordrep = WordRep(data)

        # Calculate input dimension for the main BiLSTM:
        # word_emb_dim + global_hidden_dim (from sentence-level features)
        self.input_size = data.word_emb_dim + data.global_hidden_dim
        if self.use_char:
            # Add IntNet character feature dimension
            kernel_type = data.HP_intNet_kernel_type
            char_dim = data.HP_char_hidden_dim
            self.input_size += int( (data.HP_intNet_layer - 1) // 2 * char_dim * kernel_type + char_dim * 2 * kernel_type)
        
        # BiLSTM: split hidden dim in half for each direction
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        # Main sequence feature extractor (LSTM or GRU)
        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        
        # Document-level components
        self.mem_alpha = data.mem_bank_alpha  # Interpolation weight: (1-α)*word + α*document
        self.mem_bank = MemoryBank(data)      # Memory bank for document context
        self.mem2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)  # Project memory output to label space
       
        # Sentence-level components
        # 1D CNN over label similarity scores for refinement
        self.label2cnn = nn.Conv1d(self.label_alphabet_size, self.label_alphabet_size, kernel_size=data.global_kernel_size, padding=data.global_kernel_size//2)
        self.sentrep = SentenceRep(data)  # Sentence-level feature extractor
       
        # Linear projection: BiLSTM hidden -> label scores
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)
        
        # Move to GPU if available
        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.mem2tag = self.mem2tag.cuda()  
            self.lstm = self.lstm.cuda()
            self.mem_bank = self.mem_bank.cuda()
            self.label2cnn = self.label2cnn.cuda()

    def partial_softmax(self, inputs, mask, dim):
        """
        Masked softmax: compute softmax only over valid (non-padded) positions.
        
        Args:
            inputs: (batch_size, sent_len) - Raw scores.
            mask: (batch_size, sent_len) - Binary mask.
            dim: Dimension to normalize along.
        """
        exp_inp = torch.exp(inputs)
        exp_inp_weighted = torch.mul(exp_inp, mask.float())
        exp_inp_sum = torch.sum(exp_inp_weighted, dim=dim, keepdim=True)
        partial_softmax_score = torch.div(exp_inp_weighted, exp_inp_sum)
        return partial_softmax_score


    def get_sentence_embedd(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, label_embs, mask):
        """
        Compute sentence-level representation using label-attention.
        
        Steps:
        1. Extract sentence-level features via SentenceRep (separate BiLSTM/CNN)
        2. Refine label-similarity scores using 1D CNN
        3. Compute attention weights from refined label scores (masked softmax)
        4. Weighted sum of sentence features -> sentence representation
        5. Repeat for each word position (same sentence context for all words)
        
        Args:
            label_embs: (batch, sent_len, num_labels) - Cosine similarity scores.
            mask: (batch, sent_len) - Binary mask.
            
        Returns:
            sentence_represent: (batch, sent_len, global_hidden_dim) - Sentence context.
        """
        batch_size, seq_len = word_inputs.size()
        # Get global sentence features: (batch, sent_len, global_hidden_dim)
        sentence_hidden = self.sentrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # Refine label similarity via 1D CNN: (batch, num_labels, sent_len) -> ReLU -> transpose back
        label_feature = F.relu(self.label2cnn(label_embs.transpose(2,1).contiguous())).transpose(2,1).contiguous()
        # Max-pool over label dimension to get per-position attention scores
        label_feature_max, _ = torch.max(label_feature, dim=-1)
        # Masked softmax to get attention weights over word positions
        label_feature_max = self.partial_softmax(label_feature_max, mask, dim=1)
        # Weighted combination: attention over sentence features -> single sentence vector
        # Then repeat for all positions: (batch, 1, global_dim) -> (batch, sent_len, global_dim)
        sentence_represent = torch.bmm(label_feature_max.unsqueeze(1), sentence_hidden).repeat(1, seq_len, 1) 
        return sentence_represent


    def get_document_embedd(self, word_inputs, hidden_embs, word_embs, idx_inputs, mask):
        """
        Retrieve document-level context from the memory bank.
        
        Flattens batch+sequence dimensions, queries the memory bank for each word,
        and reshapes back to (batch, sent_len, hidden_dim). Also updates the bank
        with the current hidden states.
        
        Args:
            word_inputs: (batch, sent_len) - Word indices.
            hidden_embs: (batch, sent_len, hidden_dim) - Current BiLSTM outputs.
            word_embs: (batch, sent_len, word_dim) - Word embeddings.
            idx_inputs: (batch, sent_len) - Global word instance indices.
            mask: (batch, sent_len) - Valid position mask.
            
        Returns:
            document_represent: (batch, sent_len, label_size) - Document-level scores.
        """
        batch_size, seq_len = word_inputs.size()
        idx_inputs = torch.masked_select(idx_inputs, mask)
        word_inputs = word_inputs.unsqueeze(-1)

        total_len = len(idx_inputs)
        hidden_dim, word_dim = hidden_embs.size(-1), word_embs.size(-1)

        # Flatten batch and sequence dimensions for memory bank query
        hidden_inp = torch.zeros(total_len, hidden_dim).type_as(hidden_embs)
        word_inp = torch.zeros(total_len, word_dim).type_as(word_embs)
        inp = torch.zeros(total_len, 1).type_as(word_inputs)

        start_idx = 0
        for i in range(batch_size):
            hidden_inp[start_idx: start_idx + mask[i].sum()] = hidden_embs[i][:mask[i].sum()]
            word_inp[start_idx: start_idx + mask[i].sum()] = word_embs[i][:mask[i].sum()]
            inp[start_idx: start_idx + mask[i].sum()] = word_inputs[i][:mask[i].sum()]
            start_idx = start_idx + mask[i].sum()
       
        # Query memory bank for document-level context
        document_hidden = self.mem_bank(inp, word_inp)
        # Update memory bank with current batch's representations
        self.mem_bank.update(idx_inputs, word_inp, hidden_inp)

        # Reshape back to (batch, sent_len, hidden_dim)
        document_represent = torch.zeros_like(hidden_embs) 
        start_idx = 0 
        for idx in range(batch_size):
            document_represent[i, :mask[i].sum()] = document_hidden[start_idx: start_idx+mask[i].sum()]
            start_idx = start_idx + mask[i].sum()
        # Project from hidden space to label space
        document_represent = self.mem2tag(document_represent)

        return document_represent



    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, idx_inputs=None):
        """
        Full hierarchical forward pass combining word, sentence, and document levels.
        
        Pipeline:
        1. WordRep: Get word+char embeddings and label-similarity scores
        2. SentenceRep + label-attention: Get sentence-level context
        3. Concatenate word+sentence features
        4. BiLSTM: Process combined features
        5. hidden2tag: Project to label emission scores
        6. MemoryBank: Get document-level context and interpolate
        
        Args:
            word_inputs: (batch_size, sent_len) - Word indices.
            word_seq_lengths: (batch_size,) - Sequence lengths.
            char_inputs: (batch_size*sent_len, word_length) - Character indices.
            char_seq_lengths: Character lengths.
            char_seq_recover: Character order recovery indices.
            mask: (batch_size, sent_len) - Valid position mask.
            idx_inputs: (batch_size, sent_len) - Global word instance indices.
            
        Returns:
            outputs: (batch_size, sent_len, label_size) - Emission scores for each label.
        """
        batch_size, seq_len = word_inputs.size()

        # Step 1: Word-level representations (word + char + label similarity)
        word_represent, label_embs, word_embs = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        
        # Step 2: Sentence-level context via label-attention
        sentence_represent = self.get_sentence_embedd(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, label_embs, mask)
        
        # Step 3: Concatenate word-level and sentence-level features
        word_represent = torch.cat([word_represent, sentence_represent], 2)

        # Step 4: BiLSTM processing (with packed sequences for efficiency)
        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        feature_out = self.droplstm(lstm_out.transpose(1,0).contiguous())

        # Step 5: Project BiLSTM output to label emission scores
        outputs = self.hidden2tag(feature_out)

        # Step 6: Document-level context via memory bank and interpolation
        doc_represent = self.get_document_embedd(word_inputs, feature_out, word_embs, idx_inputs, mask)            
        # Interpolate: (1 - α) * word_level_scores + α * document_level_scores
        outputs = outputs * (1 - self.mem_alpha) + doc_represent * self.mem_alpha

        return outputs 
