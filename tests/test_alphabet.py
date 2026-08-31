"""
Alphabet: the bidirectional token <-> index map shared by words, characters and
labels. Index 0 is reserved for padding, so real entries start at 1 and size()
counts one more than the number of stored instances.
"""
import pytest

from utils.alphabet import Alphabet


class TestIndexing:
    def test_word_alphabets_are_seeded_with_pad_and_unknown(self):
        a = Alphabet("word")
        assert a.get_index("<PAD>") == 1
        assert a.get_index("</unk>") == 2
        assert a.size() == 3  # PAD, UNK, plus the reserved index 0

    def test_label_alphabets_are_not_seeded(self):
        a = Alphabet("label", label=True)
        assert a.size() == 1  # only the reserved index 0
        assert a.get_index("B-PER") == 1

    def test_indices_start_at_one_and_increment(self):
        a = Alphabet("label", label=True)
        assert [a.get_index(t) for t in ["B-PER", "E-PER", "O"]] == [1, 2, 3]

    def test_repeated_tokens_reuse_the_same_index(self):
        a = Alphabet("label", label=True)
        assert a.get_index("B-PER") == a.get_index("B-PER")
        assert a.size() == 2

    def test_add_is_idempotent(self):
        a = Alphabet("label", label=True)
        a.add("O")
        a.add("O")
        assert a.size() == 2


class TestRoundTrip:
    def test_index_then_instance_returns_the_original_token(self):
        a = Alphabet("label", label=True)
        for token in ["B-PER", "I-PER", "E-PER", "S-LOC", "O"]:
            assert a.get_instance(a.get_index(token)) == token

    def test_index_zero_maps_to_the_first_entry(self):
        # Index 0 is the padding slot and has no token of its own, so the lookup
        # falls back to the first real entry.
        a = Alphabet("label", label=True)
        a.add("O")
        assert a.get_instance(0) == "O"


class TestGrowing:
    def test_an_open_alphabet_admits_new_tokens(self):
        a = Alphabet("word")
        before = a.size()
        assert a.get_index("Brussels") == before
        assert a.size() == before + 1

    def test_a_closed_alphabet_maps_new_tokens_to_unknown(self):
        a = Alphabet("word")
        known = a.get_index("Brussels")
        a.close()
        size_before = a.size()
        assert a.get_index("Antwerp") == a.get_index("</unk>")
        assert a.size() == size_before  # nothing was added
        assert a.get_index("Brussels") == known  # known tokens still resolve

    def test_reopening_admits_tokens_again(self):
        a = Alphabet("word")
        a.close()
        a.open()
        size_before = a.size()
        a.get_index("Antwerp")
        assert a.size() == size_before + 1

    def test_a_closed_label_alphabet_raises_on_unknown_labels(self):
        # Label alphabets have no UNKNOWN entry, so an unseen label at inference
        # time is a hard error rather than a silent mismatch.
        a = Alphabet("label", label=True)
        a.add("O")
        a.close()
        with pytest.raises(KeyError):
            a.get_index("B-PER")


class TestClearAndPersistence:
    def test_clear_empties_the_alphabet(self):
        a = Alphabet("word")
        a.get_index("Brussels")
        a.clear()
        assert a.size() == 1
        assert a.instances == []

    def test_content_round_trips_through_json(self, tmp_path):
        a = Alphabet("word")
        for token in ["EU", "rejects", "German"]:
            a.get_index(token)
        a.save(str(tmp_path))

        b = Alphabet("word")
        b.clear()
        b.load(str(tmp_path))
        assert b.instances == a.instances
        assert b.instance2index == a.instance2index
        assert b.get_instance(b.get_index("German")) == "German"


class TestEnumeration:
    def test_iteritems_yields_every_token_with_its_index(self):
        a = Alphabet("label", label=True)
        for token in ["B-PER", "O"]:
            a.get_index(token)
        assert dict(a.iteritems()) == {"B-PER": 1, "O": 2}

    def test_enumerate_items_pairs_indices_with_tokens(self):
        a = Alphabet("label", label=True)
        for token in ["B-PER", "E-PER", "O"]:
            a.get_index(token)
        assert list(a.enumerate_items()) == [(1, "B-PER"), (2, "E-PER"), (3, "O")]

    def test_enumerate_items_rejects_an_out_of_range_start(self):
        a = Alphabet("label", label=True)
        a.add("O")
        with pytest.raises(IndexError):
            a.enumerate_items(start=0)
