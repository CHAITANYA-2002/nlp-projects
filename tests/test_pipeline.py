"""
End-to-end checks on the training pipeline.

The original code targeted PyTorch 0.4.1. Several things it relied on were later
removed or tightened, so these tests exist to prove the pipeline still builds a
model, computes a loss, takes a gradient step, saves a checkpoint and decodes
with it on the PyTorch version actually installed.

Everything runs on the four sentences in sample_data/eng.bioes.train with tiny
dimensions, so the whole module takes a few seconds.
"""
import pathlib

import numpy as np
import pytest
import torch

from main import batchify_with_label, evaluate, train
from model.seqlabel import SeqLabel
from utils.data import Data

SAMPLE = str(pathlib.Path(__file__).resolve().parent.parent / "sample_data" / "eng.bioes.train")


def make_data(char_emb_dim=16, char_hidden_dim=8, intNet_layer=5, seed=42):
    """A minimal Data object wired to the sample file, with no pretrained vectors."""
    # Word, character and label embedding tables are initialised with
    # numpy.random.uniform, so pinning torch's generator alone is not enough to
    # make model construction reproducible.
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = Data()
    data.train_dir = data.dev_dir = data.test_dir = SAMPLE
    data.word_emb_dir = None
    data.label_emb_dir = None
    data.HP_gpu = False
    data.word_emb_dim = 24
    data.char_emb_dim = char_emb_dim
    data.HP_char_hidden_dim = char_hidden_dim
    data.HP_intNet_layer = intNet_layer
    data.HP_hidden_dim = 32
    data.global_hidden_dim = 16
    data.HP_batch_size = 2
    data.HP_iteration = 1

    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.fix_alphabet()
    data.generate_instance("train")
    data.generate_instance("dev")
    data.generate_instance("test")
    return data


def first_batch(data):
    return batchify_with_label(data.train_Ids[:2], False, True, False)


class TestDataLoading:
    def test_the_sample_file_yields_four_sentences(self):
        data = make_data()
        assert len(data.train_Ids) == 4

    def test_every_label_in_the_sample_reaches_the_alphabet(self):
        data = make_data()
        labels = set(data.label_alphabet.instances)
        assert {"O", "S-ORG", "S-MISC", "B-PER", "E-PER", "S-LOC"} <= labels

    def test_alphabets_are_frozen_after_fixing(self):
        data = make_data()
        assert data.word_alphabet.keep_growing is False
        assert data.label_alphabet.keep_growing is False


class TestForwardAndBackward:
    def test_the_model_builds(self):
        model = SeqLabel(make_data())
        assert isinstance(model, torch.nn.Module)
        assert any(p.requires_grad for p in model.parameters())

    def test_loss_is_a_finite_positive_scalar(self):
        data = make_data()
        model = SeqLabel(data)
        bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
        loss, tag_seq = model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)
        assert torch.isfinite(loss)
        assert float(loss) > 0

    def test_decoded_tags_have_one_label_per_token(self):
        data = make_data()
        model = SeqLabel(data)
        bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
        _, tag_seq = model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)
        assert tag_seq.shape == bw.shape

    def test_gradients_reach_every_named_component(self):
        data = make_data()
        model = SeqLabel(data)
        bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
        model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)[0].backward()

        touched = {name for name, p in model.named_parameters()
                   if p.grad is not None and torch.any(p.grad != 0)}
        for component in ["word_hidden.wordrep.word_embedding",
                          "word_hidden.wordrep.char_feature",
                          "word_hidden.lstm",
                          "crf.transitions"]:
            assert any(name.startswith(component) for name in touched), component

    def test_one_optimiser_step_reduces_the_loss_on_a_repeated_batch(self):
        data = make_data()
        model = SeqLabel(data)
        batch = first_batch(data)
        bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = batch
        optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

        model.train()
        before = float(model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)[0])
        for _ in range(5):
            optimiser.zero_grad()
            loss, _ = model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)
            loss.backward()
            optimiser.step()
        after = float(model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)[0])
        assert after < before

    def test_evaluation_runs_without_gradients_and_returns_the_metric_tuple(self):
        data = make_data()
        model = SeqLabel(data)
        speed, acc, p, r, f, results, scores = evaluate(data, model, "dev")
        assert 0.0 <= acc <= 1.0
        assert len(results) == len(data.dev_Ids)


class TestConfigurationRobustness:
    @pytest.mark.parametrize("char_emb_dim,char_hidden_dim", [
        (16, 8),    # char_emb_dim == 2 * char_hidden_dim, the shipped ratio
        (8, 8),     # equal
        (8, 16),    # embedding narrower than the hidden size
        (32, 4),
    ])
    def test_the_model_trains_for_char_dimensions_other_than_the_shipped_ratio(
            self, char_emb_dim, char_hidden_dim):
        # Downstream layers used to assume char_emb_dim == 2 * char_hidden_dim and
        # failed with an opaque matmul shape error otherwise.
        data = make_data(char_emb_dim=char_emb_dim, char_hidden_dim=char_hidden_dim)
        model = SeqLabel(data)
        bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
        loss, _ = model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)
        loss.backward()
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("intNet_layer", [3, 5, 7])
    def test_the_model_trains_at_several_intnet_depths(self, intNet_layer):
        data = make_data(intNet_layer=intNet_layer)
        model = SeqLabel(data)
        bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
        assert torch.isfinite(model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)[0])

    def test_the_bilstm_input_width_matches_what_wordrep_emits(self):
        data = make_data(char_emb_dim=8, char_hidden_dim=16)
        model = SeqLabel(data)
        sequence = model.word_hidden
        assert sequence.input_size == sequence.wordrep.output_dim + data.global_hidden_dim
        assert sequence.wordrep.output_dim == (
            data.word_emb_dim + sequence.wordrep.char_feature.output_dim)


class TestDeterminism:
    def test_seeding_torch_alone_is_not_enough(self):
        # Documents why make_data seeds numpy too: the embedding tables come from
        # numpy's global stream, which keeps advancing between constructions.
        def loss_with_torch_seed_only():
            torch.manual_seed(7)
            data = Data()
            data.train_dir = data.dev_dir = data.test_dir = SAMPLE
            data.word_emb_dir = None
            data.label_emb_dir = None
            data.HP_gpu = False
            data.word_emb_dim = 24
            data.char_emb_dim = 16
            data.HP_char_hidden_dim = 8
            data.HP_intNet_layer = 5
            data.HP_hidden_dim = 32
            data.global_hidden_dim = 16
            data.initial_feature_alphabets()
            data.build_alphabet(data.train_dir)
            data.fix_alphabet()
            data.generate_instance("train")
            model = SeqLabel(data)
            model.eval()
            bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
            with torch.no_grad():
                return float(model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)[0])

        np.random.seed(0)
        assert loss_with_torch_seed_only() != loss_with_torch_seed_only()

    def test_the_same_seed_gives_the_same_loss(self):
        losses = []
        for _ in range(2):
            data = make_data(seed=7)
            model = SeqLabel(data)
            bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
            model.eval()
            with torch.no_grad():
                losses.append(float(model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)[0]))
        assert losses[0] == pytest.approx(losses[1])

    def test_different_seeds_give_different_losses(self):
        def loss_for(seed):
            data = make_data(seed=seed)
            model = SeqLabel(data)
            bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
            model.eval()
            with torch.no_grad():
                return float(model.calculate_loss(bw, bwl, bc, bcl, bcr, bl, mask, bidx)[0])
        assert loss_for(1) != loss_for(2)


class TestCheckpointRoundTrip:
    def test_training_writes_a_checkpoint_and_a_dataset_file(self, tmp_path):
        data = make_data()
        data.model_dir = str(tmp_path / "nested" / "run")
        train(data)
        # The parent directory did not exist beforehand; saving must create it.
        assert (tmp_path / "nested" / "run.dset").exists()
        assert (tmp_path / "nested" / "run.0.model").exists()

    def test_a_reloaded_checkpoint_decodes_identically(self, tmp_path):
        data = make_data()
        model = SeqLabel(data)
        model.eval()
        bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
        with torch.no_grad():
            before = model(bw, bwl, bc, bcl, bcr, mask, bidx)

        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        restored = SeqLabel(data)
        restored.load_state_dict(
            torch.load(path, map_location=torch.device("cpu"), weights_only=True))
        restored.eval()
        with torch.no_grad():
            after = restored(bw, bwl, bc, bcl, bcr, mask, bidx)
        assert torch.equal(before, after)


class TestMasking:
    def test_the_mask_is_boolean(self):
        # masked_select and friends stopped accepting uint8 masks, so the mask
        # dtype is load-bearing rather than cosmetic.
        data = make_data()
        mask = first_batch(data)[-1]
        assert mask.dtype == torch.bool

    def test_the_mask_marks_exactly_the_real_tokens(self):
        data = make_data()
        bw, bf, bwl, bwr, bc, bcl, bcr, bl, bidx, mask = first_batch(data)
        assert mask.sum(dim=1).tolist() == bwl.tolist()

    def test_batches_are_sorted_by_descending_length(self):
        data = make_data()
        lengths = first_batch(data)[2].tolist()
        assert lengths == sorted(lengths, reverse=True)
