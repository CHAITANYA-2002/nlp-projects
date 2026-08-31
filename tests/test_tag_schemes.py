"""
Tag scheme conversion between IOB, BIO and BIOES.

The converters are file-in/file-out, so each test writes a small CoNLL file to a
tmp_path and reads the result back as (token, tag) pairs.
"""
import pytest

from utils.tagSchemeConverter import BIO2BIOES, BIOES2BIO, IOB2BIO


def write(path, rows, trailing_blank=True):
    """rows is a list of sentences; each sentence is a list of 'token TAG' strings."""
    text = "\n\n".join("\n".join(sent) for sent in rows)
    if trailing_blank:
        text += "\n\n"
    path.write_text(text, encoding="utf-8")
    return str(path)


def read(path):
    sentences, current = [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        token, tag = line.split()
        current.append((token, tag))
    if current:
        sentences.append(current)
    return sentences


def convert(fn, tmp_path, rows, trailing_blank=True):
    src = write(tmp_path / "in.txt", rows, trailing_blank)
    dst = str(tmp_path / "out.txt")
    fn(src, dst)
    return read(dst)


class TestBIO2BIOES:
    def test_lone_begin_becomes_single(self, tmp_path):
        out = convert(BIO2BIOES, tmp_path, [["EU B-ORG", "rejects O"]])
        assert out == [[("EU", "S-ORG"), ("rejects", "O")]]

    def test_final_inside_becomes_end(self, tmp_path):
        out = convert(BIO2BIOES, tmp_path, [["Peter B-PER", "Blackburn I-PER"]])
        assert out == [[("Peter", "B-PER"), ("Blackburn", "E-PER")]]

    def test_three_token_entity_keeps_a_middle_inside_tag(self, tmp_path):
        out = convert(BIO2BIOES, tmp_path, [["a B-ORG", "b I-ORG", "c I-ORG"]])
        assert [tag for _, tag in out[0]] == ["B-ORG", "I-ORG", "E-ORG"]

    def test_outside_tags_pass_through_untouched(self, tmp_path):
        out = convert(BIO2BIOES, tmp_path, [["x O", "y O"]])
        assert [tag for _, tag in out[0]] == ["O", "O"]

    def test_entity_at_the_very_end_of_a_sentence_is_closed(self, tmp_path):
        out = convert(BIO2BIOES, tmp_path, [["a O", "b B-LOC", "c I-LOC"]])
        assert [tag for _, tag in out[0]] == ["O", "B-LOC", "E-LOC"]

    def test_the_last_sentence_survives_without_a_trailing_blank_line(self, tmp_path):
        # Sentences are flushed when a blank line is read, so a file that ends
        # on a token line used to lose its final sentence entirely.
        out = convert(BIO2BIOES, tmp_path,
                      [["a B-ORG"], ["b B-LOC"]], trailing_blank=False)
        assert len(out) == 2
        assert out[1] == [("b", "S-LOC")]


class TestBIOES2BIO:
    def test_single_becomes_begin(self, tmp_path):
        out = convert(BIOES2BIO, tmp_path, [["EU S-ORG"]])
        assert out == [[("EU", "B-ORG")]]

    def test_end_becomes_inside(self, tmp_path):
        out = convert(BIOES2BIO, tmp_path, [["Peter B-PER", "Blackburn E-PER"]])
        assert [tag for _, tag in out[0]] == ["B-PER", "I-PER"]

    def test_begin_and_inside_are_left_alone(self, tmp_path):
        out = convert(BIOES2BIO, tmp_path, [["a B-ORG", "b I-ORG", "c E-ORG"]])
        assert [tag for _, tag in out[0]] == ["B-ORG", "I-ORG", "I-ORG"]


class TestRoundTrip:
    @pytest.mark.parametrize("tags", [
        ["B-ORG"],
        ["B-PER", "I-PER"],
        ["O", "B-LOC", "I-LOC", "O"],
        ["B-ORG", "B-ORG"],
        ["B-MISC", "O", "B-MISC", "I-MISC"],
    ])
    def test_bio_survives_a_trip_through_bioes(self, tmp_path, tags):
        rows = [[f"w{i} {tag}" for i, tag in enumerate(tags)]]
        src = write(tmp_path / "a.txt", rows)
        mid, dst = str(tmp_path / "b.txt"), str(tmp_path / "c.txt")
        BIO2BIOES(src, mid)
        BIOES2BIO(mid, dst)
        assert [tag for _, tag in read(dst)[0]] == tags


class TestIOB2BIO:
    def test_iob_inside_that_opens_an_entity_is_promoted_to_begin(self, tmp_path):
        # IOB1 only uses B- to split two adjacent same-type entities, so an
        # entity-initial token normally carries I-.
        out = convert(IOB2BIO, tmp_path, [["EU I-ORG", "rejects O"]])
        assert out[0][0] == ("EU", "B-ORG")

    def test_continuation_stays_inside(self, tmp_path):
        out = convert(IOB2BIO, tmp_path, [["Peter I-PER", "Blackburn I-PER"]])
        assert [tag for _, tag in out[0]] == ["B-PER", "I-PER"]

    def test_type_change_starts_a_new_entity(self, tmp_path):
        out = convert(IOB2BIO, tmp_path, [["a I-PER", "b I-LOC"]])
        assert [tag for _, tag in out[0]] == ["B-PER", "B-LOC"]
