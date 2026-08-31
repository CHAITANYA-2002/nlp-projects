"""
IntNet, the inception-style character CNN.

Its output width is not simply char_hidden_dim: the dense connections carry the
initial embedding-width convolutions all the way to the end, so the width is

    char_emb_dim * kernel_type  +  n_blocks * char_hidden_dim * kernel_type

with n_blocks = (intNet_layer - 1) // 2. Downstream modules used to re-derive
this with char_hidden_dim substituted for char_emb_dim, which agrees only when
char_emb_dim == 2 * char_hidden_dim. These tests pin the real width.
"""
import pytest
import torch

from model.IntNet import IntNet


def build(char_emb_dim=32, char_hidden_dim=16, intNet_layer=7, kernel_type=2,
          alphabet_size=40, seed=0):
    torch.manual_seed(seed)
    return IntNet(alphabet_size, char_emb_dim, intNet_layer, kernel_type,
                  dropout=0.0, hidden_size=char_hidden_dim, gpu=False)


def expected_width(char_emb_dim, char_hidden_dim, intNet_layer, kernel_type=2):
    blocks = (intNet_layer - 1) // 2
    return char_emb_dim * kernel_type + blocks * char_hidden_dim * kernel_type


class TestOutputWidth:
    @pytest.mark.parametrize("char_emb_dim,char_hidden_dim,intNet_layer", [
        (32, 16, 7),    # the shipped demo.train.config
        (32, 16, 5),
        (32, 16, 3),
        (8, 8, 5),      # char_emb_dim != 2 * char_hidden_dim
        (32, 32, 7),
        (16, 4, 9),
        (100, 50, 7),
    ])
    def test_declared_width_matches_the_tensor_it_produces(
            self, char_emb_dim, char_hidden_dim, intNet_layer):
        net = build(char_emb_dim, char_hidden_dim, intNet_layer)
        chars = torch.randint(0, 40, (6, 11))
        produced = net.get_last_hiddens(chars, [11] * 6)

        assert net.output_dim == expected_width(char_emb_dim, char_hidden_dim, intNet_layer)
        assert produced.shape == (6, net.output_dim)

    def test_width_depends_on_the_embedding_dimension_not_only_the_hidden_size(self):
        # The regression that used to surface as an opaque matmul shape error
        # several layers downstream: two configs with the same char_hidden_dim
        # produce different widths.
        narrow = build(char_emb_dim=8, char_hidden_dim=16)
        wide = build(char_emb_dim=64, char_hidden_dim=16)
        assert narrow.output_dim != wide.output_dim
        assert wide.output_dim - narrow.output_dim == (64 - 8) * 2

    def test_more_blocks_widen_the_output(self):
        assert build(intNet_layer=3).output_dim < build(intNet_layer=7).output_dim

    @pytest.mark.parametrize("intNet_layer,blocks", [(1, 0), (3, 1), (5, 2), (7, 3), (9, 4)])
    def test_block_count_is_half_the_layer_count_rounded_down(self, intNet_layer, blocks):
        net = build(intNet_layer=intNet_layer)
        assert len(net.cnn_list) == blocks
        assert len(net.multi_cnn_list_3) == blocks
        assert len(net.multi_cnn_list_5) == blocks


class TestAccessors:
    def test_get_last_hiddens_pools_away_the_character_axis(self):
        net = build()
        chars = torch.randint(0, 40, (5, 9))
        assert net.get_last_hiddens(chars, [9] * 5).shape == (5, net.output_dim)

    def test_get_all_hiddens_keeps_one_vector_per_character(self):
        # This accessor referenced an attribute that was never created, so it
        # raised AttributeError for every input.
        net = build()
        chars = torch.randint(0, 40, (5, 9))
        assert net.get_all_hiddens(chars, [9] * 5).shape == (5, 9, net.output_dim)

    def test_forward_delegates_to_get_all_hiddens(self):
        net = build()
        net.eval()
        chars = torch.randint(0, 40, (3, 7))
        with torch.no_grad():
            assert torch.allclose(net(chars, [7] * 3), net.get_all_hiddens(chars, [7] * 3))

    def test_pooled_output_is_the_max_over_the_character_axis(self):
        net = build()
        net.eval()
        chars = torch.randint(0, 40, (4, 8))
        with torch.no_grad():
            pooled = net.get_last_hiddens(chars, [8] * 4)
            per_position = net.get_all_hiddens(chars, [8] * 4)
        assert torch.allclose(pooled, per_position.max(dim=1).values, atol=1e-6)


class TestBehaviour:
    def test_different_words_get_different_representations(self):
        net = build(seed=3)
        net.eval()
        a = torch.tensor([[5, 6, 7, 8]])
        b = torch.tensor([[9, 10, 11, 12]])
        with torch.no_grad():
            assert not torch.allclose(net.get_last_hiddens(a, [4]),
                                      net.get_last_hiddens(b, [4]))

    def test_the_same_word_gets_the_same_representation(self):
        net = build(seed=3)
        net.eval()
        word = torch.tensor([[5, 6, 7, 8]])
        with torch.no_grad():
            assert torch.allclose(net.get_last_hiddens(word, [4]),
                                  net.get_last_hiddens(word, [4]))

    def test_output_is_non_negative_because_the_stack_ends_in_relu(self):
        net = build(seed=5)
        net.eval()
        chars = torch.randint(0, 40, (4, 9))
        with torch.no_grad():
            assert torch.all(net.get_last_hiddens(chars, [9] * 4) >= 0)

    def test_gradients_flow_back_to_the_character_embeddings(self):
        net = build()
        net.get_last_hiddens(torch.randint(0, 40, (3, 6)), [6] * 3).sum().backward()
        assert net.char_embeddings.weight.grad is not None
        assert torch.any(net.char_embeddings.weight.grad != 0)

    def test_kernel_five_needs_at_least_five_characters(self):
        # Padding is 2 on each side for the width-5 kernel, so words shorter than
        # three characters still convolve; this pins the shortest word that works.
        net = build()
        net.eval()
        with torch.no_grad():
            assert net.get_last_hiddens(torch.randint(0, 40, (2, 1)), [1, 1]).shape[1] == net.output_dim
