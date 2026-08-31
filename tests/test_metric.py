"""
Entity-span extraction and the precision/recall/F1 that main.py reports.

Spans are encoded as strings of the form "[start,end]TYPE" (or "[i]TYPE" for a
single token), because get_ner_fmeasure compares gold and predicted spans with
set intersection and needs them hashable.
"""
import pytest

from utils.metric import get_ner_BMES, get_ner_BIO, get_ner_fmeasure, reverse_style


class TestReverseStyle:
    def test_moves_the_bracket_group_to_the_front(self):
        assert reverse_style("PER[0,1]") == "[0,1]PER"

    def test_is_idempotent_once_the_bracket_leads(self):
        # The bracket group is found by index, so a string that already starts
        # with '[' is returned unchanged rather than rotated back.
        assert reverse_style("[0,1]PER") == "[0,1]PER"


class TestBIOESSpans:
    def test_single_token_entity(self):
        assert get_ner_BMES(["S-ORG", "O"]) == ["[0]ORG"]

    def test_two_token_entity(self):
        assert get_ner_BMES(["B-PER", "E-PER", "O"]) == ["[0,1]PER"]

    def test_three_token_entity_spans_begin_to_end(self):
        assert get_ner_BMES(["B-PER", "I-PER", "E-PER"]) == ["[0,2]PER"]

    def test_sentence_with_no_entities(self):
        assert get_ner_BMES(["O", "O", "O"]) == []

    def test_two_entities_are_kept_separate(self):
        assert get_ner_BMES(["S-ORG", "O", "B-LOC", "E-LOC"]) == ["[0]ORG", "[2,3]LOC"]

    def test_entity_opened_but_never_closed_is_recorded_as_a_point_span(self):
        # A B- with no matching E- is malformed. Rather than dropping it, the
        # extractor emits a start-only span, which can never match a well-formed
        # gold span and so counts as a false positive.
        assert get_ner_BMES(["B-PER", "O", "O"]) == ["[0]PER"]

    def test_labels_are_case_insensitive(self):
        assert get_ner_BMES(["s-org"]) == get_ner_BMES(["S-ORG"])


class TestBIOSpans:
    def test_entity_then_gap_then_entity(self):
        assert get_ner_BIO(["B-PER", "I-PER", "O", "B-LOC"]) == ["[0,1]PER", "[3]LOC"]

    def test_adjacent_entities_of_the_same_type_are_split_at_the_second_B(self):
        # This is the reason BIOES is preferred: in BIO, back-to-back entities of
        # the same type are only separable because a second B- appears.
        assert get_ner_BIO(["B-PER", "B-PER"]) == ["[0,0]PER", "[1]PER"]

    def test_inside_tag_with_no_preceding_begin_is_dropped(self):
        assert get_ner_BIO(["I-PER", "O"]) == []

    def test_type_change_mid_entity_closes_the_first(self):
        assert get_ner_BIO(["B-PER", "I-LOC"]) == ["[0,0]PER"]


class TestFMeasure:
    def test_perfect_prediction(self):
        acc, p, r, f = get_ner_fmeasure([["B-PER", "E-PER"]], [["B-PER", "E-PER"]], "BIOES")
        assert (acc, p, r, f) == (1.0, 1.0, 1.0, 1.0)

    def test_accuracy_is_token_level_not_entity_level(self):
        # Three of four tokens are right, but the one wrong token destroys the
        # only entity. This gap is why entity-level F1 is the headline metric.
        gold = [["B-PER", "E-PER", "O", "O"]]
        pred = [["B-PER", "O", "O", "O"]]
        acc, p, r, f = get_ner_fmeasure(gold, pred, "BIOES")
        assert acc == 0.75
        assert r == 0.0

    def test_predicting_nothing_reports_precision_as_the_sentinel_minus_one(self):
        # With no predicted entities precision is undefined; the original code
        # returns -1 rather than raising, and main.py prints that value as-is.
        acc, p, r, f = get_ner_fmeasure([["B-PER", "E-PER"]], [["O", "O"]], "BIOES")
        assert p == -1
        assert r == 0.0
        assert f == -1

    def test_no_gold_entities_reports_recall_as_the_sentinel_minus_one(self):
        acc, p, r, f = get_ner_fmeasure([["O", "O"]], [["B-PER", "E-PER"]], "BIOES")
        assert p == 0.0
        assert r == -1
        assert f == -1

    def test_partial_match_precision_and_recall(self):
        gold = [["B-PER", "E-PER", "S-LOC"]]
        pred = [["B-PER", "E-PER", "S-ORG"]]
        acc, p, r, f = get_ner_fmeasure(gold, pred, "BIOES")
        assert p == 0.5
        assert r == 0.5
        assert f == 0.5

    @pytest.mark.parametrize("label_type", ["BMES", "BIOES"])
    def test_bmes_and_bioes_name_the_same_scheme(self, label_type):
        gold = [["B-PER", "E-PER"]]
        assert get_ner_fmeasure(gold, gold, label_type) == (1.0, 1.0, 1.0, 1.0)

    def test_scores_accumulate_across_sentences(self):
        gold = [["S-PER"], ["S-LOC"]]
        pred = [["S-PER"], ["O"]]
        acc, p, r, f = get_ner_fmeasure(gold, pred, "BIOES")
        assert p == 1.0
        assert r == 0.5
        assert f == pytest.approx(2 / 3)
