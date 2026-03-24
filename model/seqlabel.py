"""
seqlabel.py - Sequence Labeling Model (Top-Level)

This is the top-level model class that combines the WordSequence feature extractor
with either a CRF (Conditional Random Field) or a softmax classifier for
sequence labeling (NER). It provides two main operations:

1. calculate_loss(): Forward pass + loss computation for training
   - Uses CRF negative log-likelihood loss if use_crf=True
   - Uses NLLLoss with log-softmax if use_crf=False

2. forward(): Inference-only prediction (no loss computation)
   - Uses Viterbi decoding if CRF is enabled
   - Uses argmax over softmax scores otherwise

The label_alphabet_size is incremented by 2 to accommodate the special
START_TAG and STOP_TAG markers required by the CRF layer.
"""

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F 
from .wordsequence import WordSequence  # Main feature extractor (hierarchical representation)
from .crf import CRF  # Conditional Random Field for structured prediction


class SeqLabel(nn.Module):
    """
    Sequence Labeling model combining hierarchical feature extraction with CRF/softmax.
    
    Args:
        data: Data configuration object containing model hyperparameters,
              alphabet sizes, and embedding settings.
    
    Attributes:
        word_hidden (WordSequence): Hierarchical feature extractor producing emission scores.
        crf (CRF): Optional CRF layer for structured sequence prediction.
        use_crf (bool): Whether to use CRF (True) or softmax (False) for decoding.
        average_batch (bool): Whether to average the loss over the batch.
    """
    def __init__(self, data):
        super(SeqLabel, self).__init__()
        self.gpu = data.HP_gpu
        self.use_crf = data.use_crf  
        print("build sequence labeling network...")
        self.average_batch = data.average_batch_loss 

        # Save original label size for CRF, then add 2 for START_TAG and STOP_TAG
        # (used internally by the BiLSTM output layer)
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2

        # Build the hierarchical word sequence feature extractor
        self.word_hidden = WordSequence(data)

        # Initialize CRF layer with the original label size (before +2)
        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)
 

    def calculate_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, batch_idx=None):
        """
        Compute the training loss and predicted tag sequence.
        
        Args:
            word_inputs: (batch_size, sent_len) - Word index tensor.
            word_seq_lengths: (batch_size,) - Actual sequence lengths.
            char_inputs: (batch_size*sent_len, word_len) - Character index tensor.
            char_seq_lengths: (batch_size*sent_len,) - Character sequence lengths.
            char_seq_recover: Index to recover original character order after sorting.
            batch_label: (batch_size, sent_len) - Gold label indices.
            mask: (batch_size, sent_len) - Binary mask for valid positions (1=valid, 0=pad).
            batch_idx: Optional word index tensor for memory bank.
            
        Returns:
            total_loss: Scalar loss for backpropagation.
            tag_seq: (batch_size, sent_len) - Predicted tag sequence.
        """
        batch_size, seq_len = word_inputs.size()  
       
        # Get emission scores from the hierarchical feature extractor
        outs = self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, batch_idx)
 
        if self.use_crf:
            # CRF loss: negative log-likelihood of the gold sequence
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label).sum()
            # Viterbi decoding to get the best tag sequence
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            # Softmax loss: NLLLoss with log-softmax
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            # Argmax decoding
            _, tag_seq  = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)

        # Optionally average loss over batch
        if self.average_batch: 
            total_loss = total_loss / batch_size 
        return total_loss, tag_seq 


    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, batch_idx=None):
        """
        Forward pass for inference (prediction only, no loss).
        
        Returns:
            tag_seq: (batch_size, sent_len) - Predicted tag indices.
        """
        # Get emission scores
        outs = self.word_hidden(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, batch_idx)
        batch_size, seq_len = word_inputs.size() 

        if self.use_crf: 
            # Viterbi decoding for structured prediction
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else: 
            # Simple argmax decoding
            outs = outs.view(batch_size * seq_len, -1) 
            _, tag_seq  = torch.max(outs, 1) 
            tag_seq = tag_seq.view(batch_size, seq_len) 
            # Zero out padded positions
            tag_seq = mask.long() * tag_seq 
        return tag_seq
 
