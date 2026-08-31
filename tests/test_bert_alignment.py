"""
Sub-word to token alignment for BERT features.

BERT tokenises into word pieces, so "Blackburn" may arrive as ["Black", "##burn"]
while the NER labels are one-per-token. These reducers collapse the pieces of a
token back into a single vector.

Only get_aligned_bert_emb.py is exercised here: extract_features.py, modeling.py
and tokenization.py are Google's unmodified TensorFlow 1.x release.
"""
import importlib.util
import pathlib

import pytest

# BERT/ has no package __init__ that can be imported without TensorFlow, so the
# one pure-Python module is loaded directly by path.
_MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "BERT" / "get_aligned_bert_emb.py"
_spec = importlib.util.spec_from_file_location("get_aligned_bert_emb", _MODULE_PATH)
align = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(align)


class TestReduceMean:
    def test_single_vector_is_returned_as_is(self):
        assert align.reduce_mean_list([[1.0, 2.0, 3.0]]) == [1.0, 2.0, 3.0]

    def test_two_vectors_average_element_wise(self):
        assert align.reduce_mean_list([[0.0, 2.0], [2.0, 4.0]]) == [1.0, 3.0]

    def test_three_vectors_average_element_wise(self):
        assert align.reduce_mean_list([[3.0, 0.0], [0.0, 3.0], [3.0, 3.0]]) == [2.0, 2.0]

    def test_the_inputs_are_left_untouched(self):
        # The reducer used to accumulate into its first argument, so the caller's
        # word-piece buffer was corrupted by the act of reading it.
        pieces = [[1.0, 1.0], [3.0, 3.0]]
        align.reduce_mean_list(pieces)
        assert pieces == [[1.0, 1.0], [3.0, 3.0]]

    def test_calling_twice_gives_the_same_answer(self):
        pieces = [[1.0, 1.0], [3.0, 3.0]]
        assert align.reduce_mean_list(pieces) == align.reduce_mean_list(pieces)

    def test_an_empty_list_is_rejected(self):
        with pytest.raises(ValueError):
            align.reduce_mean_list([])


class TestReduceMax:
    def test_single_vector_is_returned_as_is(self):
        assert align.reduce_max_list([[1.0, 2.0, 3.0]]) == [1.0, 2.0, 3.0]

    def test_maximum_is_taken_per_dimension_not_per_vector(self):
        assert align.reduce_max_list([[1.0, 9.0], [9.0, 1.0]]) == [9.0, 9.0]

    def test_negative_values_are_handled(self):
        assert align.reduce_max_list([[-5.0, -1.0], [-2.0, -9.0]]) == [-2.0, -1.0]

    def test_the_inputs_are_left_untouched(self):
        pieces = [[1.0, 9.0], [9.0, 1.0]]
        align.reduce_max_list(pieces)
        assert pieces == [[1.0, 9.0], [9.0, 1.0]]

    def test_calling_twice_gives_the_same_answer(self):
        pieces = [[1.0, 9.0], [9.0, 1.0]]
        assert align.reduce_max_list(pieces) == align.reduce_max_list(pieces)

    def test_an_empty_list_is_rejected(self):
        with pytest.raises(ValueError):
            align.reduce_max_list([])


class TestReducersAgree:
    def test_both_reducers_agree_when_every_piece_is_identical(self):
        pieces = [[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]]
        assert align.reduce_mean_list(pieces) == align.reduce_max_list(pieces)

    def test_the_mean_never_exceeds_the_maximum(self):
        pieces = [[1.0, 7.0], [4.0, 2.0], [3.0, 5.0]]
        mean = align.reduce_mean_list(pieces)
        maximum = align.reduce_max_list(pieces)
        assert all(m <= x for m, x in zip(mean, maximum))


class TestArgumentParsing:
    def test_first_is_the_default_alignment_mode(self):
        args = align.parse_args(["--input_file", "in.json", "--output_file", "out.txt"])
        assert args.mode == "first"

    def test_mode_can_be_overridden(self):
        args = align.parse_args(
            ["--input_file", "in.json", "--output_file", "out.txt", "--mode", "mean"])
        assert args.mode == "mean"

    def test_the_default_delimiter_separates_tokens(self):
        args = align.parse_args(["--input_file", "in.json", "--output_file", "out.txt"])
        assert args.delimiter == "|||"
