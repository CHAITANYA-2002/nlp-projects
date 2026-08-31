"""Shared fixtures and helpers for the test suite."""
import os
import pathlib
import sys

import numpy as np
import torch

# scripts/ is not a package on the import path by default.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils.data import Data  # noqa: E402

SAMPLE = str(pathlib.Path(__file__).resolve().parent.parent
             / "sample_data" / "eng.bioes.train")


def shipped_config_data(seed=42):
    """
    A Data carrying the dimensions from demo.train.config, pointed at the sample
    sentences. Used wherever a test needs to pin a number the README quotes,
    since those numbers describe the shipped configuration.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = Data()
    data.train_dir = data.dev_dir = data.test_dir = SAMPLE
    data.word_emb_dir = None
    data.label_emb_dir = None
    data.HP_gpu = False
    data.word_emb_dim = 100
    data.char_emb_dim = 32
    data.HP_char_hidden_dim = 16
    data.HP_intNet_layer = 7
    data.HP_intNet_kernel_type = 2
    data.HP_hidden_dim = 256
    data.global_hidden_dim = 128
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.fix_alphabet()
    data.generate_instance("train")
    return data
