"""
Data reading helpers: number normalisation, pretrained embedding loading, and
turning a CoNLL file into index tensors.
"""
import numpy as np
import pytest

from utils.alphabet import Alphabet
from utils.functions import (
    normalize_word,
    norm2one,
    load_pretrain_emb,
    build_pretrain_embedding,
    read_instance,
)


class TestNormalizeWord:
    @pytest.mark.parametrize("word,expected", [
        ("1993", "0000"),
        ("B-52", "B-00"),
        ("Brussels", "Brussels"),
        ("", ""),
        ("3.14", "0.00"),
        ("1996-08-22", "0000-00-00"),
    ])
    def test_digits_collapse_to_zero(self, word, expected):
        assert normalize_word(word) == expected

    def test_every_number_of_the_same_shape_maps_to_one_token(self):
        # This is the point of the transform: it keeps the vocabulary from
        # filling up with dates and figures seen exactly once.
        assert normalize_word("1996") == normalize_word("2024")

    def test_non_ascii_digits_are_not_treated_as_digits_by_the_ascii_check(self):
        # str.isdigit() is true for superscripts and other numeric characters,
        # so they normalise too. Worth pinning: it affects the vocabulary.
        assert normalize_word("2²") == "00"


class TestNorm2One:
    def test_result_has_unit_length(self):
        v = np.array([3.0, 4.0])
        assert np.linalg.norm(norm2one(v)) == pytest.approx(1.0)

    def test_direction_is_preserved(self):
        v = np.array([3.0, 4.0])
        assert norm2one(v) == pytest.approx(np.array([0.6, 0.8]))


class TestLoadPretrainEmb:
    def write_emb(self, tmp_path, lines):
        p = tmp_path / "emb.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(p)

    def test_dimension_is_inferred_from_the_first_line(self, tmp_path):
        path = self.write_emb(tmp_path, ["the 0.1 0.2 0.3", "cat 0.4 0.5 0.6"])
        emb, dim = load_pretrain_emb(path)
        assert dim == 3
        assert sorted(emb) == ["cat", "the"]
        assert emb["the"] == pytest.approx(np.array([[0.1, 0.2, 0.3]]))

    def test_lines_with_the_wrong_width_are_skipped(self, tmp_path):
        path = self.write_emb(tmp_path, ["the 0.1 0.2", "broken 0.1 0.2 0.3", "cat 0.4 0.5"])
        emb, dim = load_pretrain_emb(path)
        assert dim == 2
        assert "broken" not in emb
        assert sorted(emb) == ["cat", "the"]

    def test_blank_lines_are_ignored(self, tmp_path):
        path = self.write_emb(tmp_path, ["the 0.1 0.2", "", "cat 0.3 0.4"])
        emb, _ = load_pretrain_emb(path)
        assert len(emb) == 2


class TestBuildPretrainEmbedding:
    def write_emb(self, tmp_path, lines):
        p = tmp_path / "emb.txt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(p)

    def test_matrix_covers_every_alphabet_entry(self, tmp_path):
        path = self.write_emb(tmp_path, ["eu 0.1 0.2", "german 0.3 0.4"])
        alphabet = Alphabet("word")
        for token in ["EU", "German", "Brussels"]:
            alphabet.get_index(token)
        matrix, dim = build_pretrain_embedding(path, alphabet, norm=False)
        assert matrix.shape == (alphabet.size(), 2)
        assert dim == 2

    def test_lookup_falls_back_to_lowercase(self, tmp_path):
        # GloVe is lowercased; the vocabulary is not. Without the case-insensitive
        # second pass, almost every capitalised entity would be out of vocabulary.
        path = self.write_emb(tmp_path, ["eu 0.1 0.2"])
        alphabet = Alphabet("word")
        alphabet.get_index("EU")
        matrix, _ = build_pretrain_embedding(path, alphabet, norm=False)
        assert matrix[alphabet.get_index("EU")] == pytest.approx([0.1, 0.2])

    def test_exact_case_wins_over_the_lowercase_fallback(self, tmp_path):
        path = self.write_emb(tmp_path, ["EU 1.0 1.0", "eu 0.1 0.2"])
        alphabet = Alphabet("word")
        alphabet.get_index("EU")
        matrix, _ = build_pretrain_embedding(path, alphabet, norm=False)
        assert matrix[alphabet.get_index("EU")] == pytest.approx([1.0, 1.0])

    def test_out_of_vocabulary_rows_are_random_but_bounded(self, tmp_path):
        path = self.write_emb(tmp_path, ["eu 0.1 0.2"])
        alphabet = Alphabet("word")
        alphabet.get_index("Antwerp")
        matrix, dim = build_pretrain_embedding(path, alphabet, norm=False)
        scale = np.sqrt(3.0 / dim)
        row = matrix[alphabet.get_index("Antwerp")]
        assert np.all(np.abs(row) <= scale)

    def test_norm_produces_unit_length_rows_for_matched_words(self, tmp_path):
        path = self.write_emb(tmp_path, ["eu 3.0 4.0"])
        alphabet = Alphabet("word")
        alphabet.get_index("eu")
        matrix, _ = build_pretrain_embedding(path, alphabet, norm=True)
        assert np.linalg.norm(matrix[alphabet.get_index("eu")]) == pytest.approx(1.0)


class TestReadInstance:
    def build(self, tmp_path, text, number_normalized=False):
        path = tmp_path / "data.txt"
        path.write_text(text, encoding="utf-8")
        word_alphabet = Alphabet("word")
        char_alphabet = Alphabet("character")
        label_alphabet = Alphabet("label", label=True)
        texts, ids, mem_mat, word_mat, word_idx = read_instance(
            str(path), word_alphabet, char_alphabet, {}, label_alphabet,
            number_normalized, -1, False, " ",
        )
        return texts, ids, word_alphabet, char_alphabet, label_alphabet

    SAMPLE = "EU S-ORG\nrejects O\n\nPeter B-PER\nBlackburn E-PER\n\n"

    def test_sentences_are_split_on_blank_lines(self, tmp_path):
        texts, ids, *_ = self.build(tmp_path, self.SAMPLE)
        assert len(texts) == 2

    def test_tokens_labels_and_characters_are_all_captured(self, tmp_path):
        # A sequence-labeling instance is [words, features, chars, labels, idxs];
        # idxs is the corpus-wide word position the memory bank keys on.
        texts, ids, *_ = self.build(tmp_path, self.SAMPLE)
        words, features, chars, labels, idxs = texts[0]
        assert words == ["EU", "rejects"]
        assert labels == ["S-ORG", "O"]
        assert chars == [["E", "U"], ["r", "e", "j", "e", "c", "t", "s"]]
        assert idxs == [1, 2]

    def test_ids_line_up_with_the_alphabets(self, tmp_path):
        texts, ids, word_alphabet, char_alphabet, label_alphabet = self.build(tmp_path, self.SAMPLE)
        words = texts[0][0]
        word_ids = ids[0][0]
        assert word_ids == [word_alphabet.get_index(w) for w in words]
        assert ids[0][3] == [label_alphabet.get_index(l) for l in texts[0][3]]

    def test_number_normalisation_changes_the_ids_but_not_the_recorded_text(self, tmp_path):
        # words.append runs before the normalisation, so the readable instance
        # keeps the original surface form while the vocabulary sees the collapsed
        # one. Decoded output therefore still shows the real token.
        text = "1996-08-22 O\n\n"
        texts, ids, word_alphabet, *_ = self.build(tmp_path, text, number_normalized=True)
        assert texts[0][0] == ["1996-08-22"]
        assert ids[0][0] == [word_alphabet.get_index("0000-00-00")]

    def test_characters_are_taken_from_the_normalised_form(self, tmp_path):
        # The character loop runs after the reassignment, so IntNet sees the
        # digit-collapsed spelling rather than the original.
        texts, *_ = self.build(tmp_path, "1996 O\n\n", number_normalized=True)
        assert texts[0][2] == [["0", "0", "0", "0"]]

    def test_a_file_with_no_blank_line_terminator_still_yields_its_sentence(self, tmp_path):
        texts, _, *_ = self.build(tmp_path, "EU S-ORG\nrejects O\n")
        assert len(texts) == 1
        assert texts[0][0] == ["EU", "rejects"]
