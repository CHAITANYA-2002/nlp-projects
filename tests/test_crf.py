"""
The CRF layer, checked against brute-force enumeration.

For a short sequence and a small tag set every possible label sequence can be
scored exhaustively, so Viterbi's argmax and the forward algorithm's partition
function both have an independent reference to be compared against. That is a
much stronger check than asserting output shapes.
"""
import itertools
import math

import pytest
import torch

from model.crf import CRF, START_TAG, STOP_TAG


def make_crf(tagset_size=3, seed=0):
    """A CRF with random (but reproducible) transition scores."""
    torch.manual_seed(seed)
    crf = CRF(tagset_size, gpu=False)
    with torch.no_grad():
        # Keep the structural -10000 blocks, jitter everything else.
        jitter = torch.randn_like(crf.transitions)
        allowed = crf.transitions > -1000
        crf.transitions.add_(jitter * allowed)
    return crf


def real_tags(tag_size):
    """Tag indices a decoded path may actually use: not padding, not START/STOP."""
    return [t for t in range(tag_size)
            if t != 0 and t != tag_size + START_TAG and t != tag_size + STOP_TAG]


def path_score(crf, feats, path):
    """Score one complete label sequence the way the CRF defines it."""
    tag_size = feats.size(-1)
    start, stop = tag_size + START_TAG, tag_size + STOP_TAG
    score = 0.0
    previous = start
    for position, tag in enumerate(path):
        score += float(crf.transitions[previous, tag]) + float(feats[position, tag])
        previous = tag
    score += float(crf.transitions[previous, stop])
    return score


def enumerate_paths(crf, feats):
    """(path, score) for every label sequence of this length."""
    tag_size = feats.size(-1)
    length = feats.size(0)
    return [(path, path_score(crf, feats, path))
            for path in itertools.product(real_tags(tag_size), repeat=length)]


class TestTransitionMatrix:
    def test_matrix_covers_the_tag_set_plus_start_and_stop(self):
        crf = CRF(5, gpu=False)
        assert crf.transitions.shape == (7, 7)

    def test_nothing_may_transition_into_start(self):
        crf = CRF(3, gpu=False)
        assert torch.all(crf.transitions[:, START_TAG] <= -10000.0)

    def test_nothing_may_transition_out_of_stop(self):
        crf = CRF(3, gpu=False)
        assert torch.all(crf.transitions[STOP_TAG, :] <= -10000.0)

    def test_the_padding_row_and_column_are_blocked(self):
        # Index 0 is the alphabet's padding slot and must never be emitted.
        crf = CRF(3, gpu=False)
        assert torch.all(crf.transitions[0, :] <= -10000.0)
        assert torch.all(crf.transitions[:, 0] <= -10000.0)

    def test_transitions_are_learnable(self):
        crf = CRF(3, gpu=False)
        assert crf.transitions.requires_grad


class TestViterbiAgainstBruteForce:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("length", [1, 2, 3])
    def test_decoded_path_is_the_highest_scoring_one(self, seed, length):
        tagset_size = 3
        crf = make_crf(tagset_size, seed=seed)
        torch.manual_seed(seed + 100)
        feats = torch.randn(1, length, tagset_size + 2)
        mask = torch.ones(1, length, dtype=torch.bool)

        _, decoded = crf._viterbi_decode(feats, mask)
        decoded_path = tuple(decoded[0].tolist())

        best_path, best_score = max(enumerate_paths(crf, feats[0]), key=lambda pair: pair[1])
        assert path_score(crf, feats[0], decoded_path) == pytest.approx(best_score, abs=1e-4)
        assert decoded_path == best_path

    def test_forward_is_the_public_entry_point_for_decoding(self):
        crf = make_crf(3, seed=7)
        feats = torch.randn(1, 3, 5)
        mask = torch.ones(1, 3, dtype=torch.bool)
        score_a, path_a = crf(feats, mask)
        score_b, path_b = crf._viterbi_decode(feats, mask)
        assert torch.equal(path_a, path_b)
        assert score_a is score_b

    def test_single_best_decoding_returns_no_path_score(self):
        # _viterbi_decode returns None in the score slot; only the n-best decoder
        # computes path scores. SeqLabel discards the slot, so nothing downstream
        # depends on it, but callers should not expect a number here.
        crf = make_crf(3, seed=7)
        feats = torch.randn(1, 3, 5)
        mask = torch.ones(1, 3, dtype=torch.bool)
        score, path = crf._viterbi_decode(feats, mask)
        assert score is None
        assert path.shape == (1, 3)

    def test_padded_positions_do_not_change_the_decoded_prefix(self):
        # Two sentences of different lengths share a batch; the shorter one is
        # padded. Its decoded labels must match decoding it on its own.
        crf = make_crf(3, seed=11)
        torch.manual_seed(3)
        feats = torch.randn(2, 4, 5)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

        _, batched = crf._viterbi_decode(feats, mask)
        _, alone = crf._viterbi_decode(feats[1:2, :2], torch.ones(1, 2, dtype=torch.bool))
        assert batched[1][:2].tolist() == alone[0].tolist()


class TestPartitionFunction:
    @pytest.mark.parametrize("length", [1, 2, 3])
    def test_log_partition_matches_brute_force_log_sum_exp(self, length):
        crf = make_crf(3, seed=5)
        torch.manual_seed(length)
        feats = torch.randn(1, length, 5)
        mask = torch.ones(1, length, dtype=torch.bool)

        forward_score, _ = crf._calculate_PZ(feats, mask)
        scores = [score for _, score in enumerate_paths(crf, feats[0])]
        expected = math.log(sum(math.exp(s) for s in scores))
        assert float(forward_score) == pytest.approx(expected, abs=1e-3)


class TestNegativeLogLikelihood:
    def test_loss_equals_log_z_minus_the_gold_path_score(self):
        crf = make_crf(3, seed=9)
        torch.manual_seed(21)
        feats = torch.randn(1, 3, 5)
        mask = torch.ones(1, 3, dtype=torch.bool)
        gold = torch.tensor([[1, 2, 1]])

        loss = crf.neg_log_likelihood_loss(feats, mask, gold)
        log_z = float(crf._calculate_PZ(feats, mask)[0])
        gold_score = path_score(crf, feats[0], tuple(gold[0].tolist()))
        assert float(loss) == pytest.approx(log_z - gold_score, abs=1e-3)

    def test_loss_is_never_negative(self):
        # log Z is a log-sum-exp over every path including the gold one, so it
        # can never be smaller than the gold path's own score.
        crf = make_crf(3, seed=13)
        torch.manual_seed(31)
        for _ in range(10):
            feats = torch.randn(2, 4, 5)
            mask = torch.ones(2, 4, dtype=torch.bool)
            gold = torch.randint(1, 4, (2, 4))
            loss = crf.neg_log_likelihood_loss(feats, mask, gold)
            assert torch.all(loss >= -1e-4)

    def test_loss_shrinks_as_the_gold_path_is_favoured(self):
        crf = make_crf(3, seed=17)
        torch.manual_seed(41)
        feats = torch.randn(1, 3, 5)
        mask = torch.ones(1, 3, dtype=torch.bool)
        gold = torch.tensor([[1, 1, 1]])

        before = float(crf.neg_log_likelihood_loss(feats, mask, gold))
        boosted = feats.clone()
        boosted[0, :, 1] += 5.0  # make the gold tag far more likely at each step
        after = float(crf.neg_log_likelihood_loss(boosted, mask, gold))
        assert after < before

    def test_gradients_reach_the_transition_matrix(self):
        crf = make_crf(3, seed=19)
        torch.manual_seed(51)
        feats = torch.randn(1, 3, 5, requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        gold = torch.tensor([[1, 2, 1]])
        crf.neg_log_likelihood_loss(feats, mask, gold).sum().backward()
        assert crf.transitions.grad is not None
        assert torch.any(crf.transitions.grad != 0)
        assert feats.grad is not None


class TestNBestDecoding:
    def test_nbest_returns_the_top_paths_in_descending_order(self):
        crf = make_crf(3, seed=23)
        torch.manual_seed(61)
        feats = torch.randn(1, 3, 5)
        mask = torch.ones(1, 3, dtype=torch.bool)
        nbest = 4

        scores, paths = crf._viterbi_decode_nbest(feats, mask, nbest)
        assert paths.shape == (1, 3, nbest)

        ranked = sorted((s for _, s in enumerate_paths(crf, feats[0])), reverse=True)
        decoded = [path_score(crf, feats[0], tuple(paths[0, :, k].tolist()))
                   for k in range(nbest)]
        assert decoded == sorted(decoded, reverse=True)
        assert decoded == pytest.approx(ranked[:nbest], abs=1e-4)

    def test_the_first_nbest_path_is_the_viterbi_path(self):
        crf = make_crf(3, seed=29)
        torch.manual_seed(71)
        feats = torch.randn(1, 4, 5)
        mask = torch.ones(1, 4, dtype=torch.bool)
        _, best = crf._viterbi_decode(feats, mask)
        _, nbest = crf._viterbi_decode_nbest(feats, mask, 3)
        assert nbest[0, :, 0].tolist() == best[0].tolist()
