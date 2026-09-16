"""Model compatibility, position encodings, attention backends and KV caches."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


SPEC = importlib.util.spec_from_file_location(
    'mini_gpt_model_test', Path(__file__).resolve().parents[1] / '05_model.py',
)
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
ATTENTION = MODEL.transformer_module.attention_module


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(12)
        self.tokens = torch.randint(0, 23, (2, 8))

    def model(self, **kwargs):
        config = dict(
            vocab_size=23, context_length=8, embedding_dim=16,
            num_heads=4, num_layers=2, ffn_dim=32, dropout=0.0,
        )
        config.update(kwargs)
        return MODEL.MiniGPT(**config)

    def test_default_state_dict_and_legacy_forward(self):
        model = self.model(dropout=0.2)
        expected = {
            'token_embedding.weight', 'position_embedding.weight',
            'final_norm.weight', 'final_norm.bias', 'lm_head.weight', 'lm_head.bias',
        }
        for layer in range(2):
            for module in (
                'attention.q_proj', 'attention.k_proj', 'attention.v_proj',
                'attention.out_proj', 'norm_1', 'norm_2',
                'feed_forward.net.0', 'feed_forward.net.2',
            ):
                expected.update(f'blocks.{layer}.{module}.{p}' for p in ('weight', 'bias'))
        self.assertEqual(set(model.state_dict()), expected)
        clone = self.model(position_encoding='learned', attention_backend='manual', dropout=0.2)
        clone.load_state_dict(model.state_dict(), strict=True)

        torch.manual_seed(42)
        actual, loss = model(self.tokens, self.tokens)
        torch.manual_seed(42)
        positions = torch.arange(8).unsqueeze(0)
        h = clone.dropout(clone.token_embedding(self.tokens) + clone.position_embedding(positions))
        for block in clone.blocks:
            attn = block.attention
            q, k, v = [
                projection(h).view(2, 8, 4, 4).transpose(1, 2)
                for projection in (attn.q_proj, attn.k_proj, attn.v_proj)
            ]
            scores = (q @ k.transpose(-2, -1)) / 2
            scores = scores.masked_fill(torch.ones(8, 8).triu(1).bool(), float('-inf'))
            output = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view(2, 8, 16)
            h = block.norm_1(h + attn.out_proj(attn.dropout(output)))
            h = block.norm_2(h + block.feed_forward(h))
        expected_logits = clone.lm_head(clone.final_norm(h))
        torch.testing.assert_close(actual, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(loss, F.cross_entropy(expected_logits.flatten(0, 1), self.tokens.flatten()))

    def test_cached_chunks_match_full_sequence(self):
        for encoding in ('learned', 'rope'):
            for backend in ('manual', 'sdpa'):
                for chunks in ((1,) * 8, (3, 2, 3)):
                    with self.subTest(encoding=encoding, backend=backend, chunks=chunks), torch.no_grad():
                        model = self.model(position_encoding=encoding, attention_backend=backend).eval()
                        full, _ = model(self.tokens)
                        cache = None
                        outputs = []
                        offset = 0
                        for size in chunks:
                            output, loss, cache = model(
                                self.tokens[:, offset:offset + size],
                                past_key_values=cache, use_cache=True,
                            )
                            outputs.append(output)
                            offset += size
                            self.assertIsNone(loss)
                            self.assertIsInstance(cache, tuple)
                            self.assertEqual(len(cache), 2)
                            for pair in cache:
                                self.assertIsInstance(pair, tuple)
                                for tensor in pair:
                                    self.assertEqual(tensor.shape, (2, 4, offset, 4))
                        torch.testing.assert_close(torch.cat(outputs, dim=1), full, atol=2e-7, rtol=1e-5)

    def test_consume_cache_without_returning_cache(self):
        model = self.model().eval()
        with torch.no_grad():
            _, _, cache = model(self.tokens[:, :3], use_cache=True)
            result = model(self.tokens[:, 3:], past_key_values=cache)
            self.assertEqual(len(result), 2)
            torch.testing.assert_close(result[0], model(self.tokens)[0][:, 3:], atol=2e-7, rtol=1e-5)

    def test_sdpa_training_dropout_and_gradients_match_manual(self):
        for encoding in ('learned', 'rope'):
            with self.subTest(encoding=encoding):
                manual = self.model(position_encoding=encoding, dropout=0.2)
                sdpa = self.model(position_encoding=encoding, attention_backend='sdpa', dropout=0.2)
                sdpa.load_state_dict(manual.state_dict())
                torch.manual_seed(7)
                expected, expected_loss = manual(self.tokens, self.tokens)
                torch.manual_seed(7)
                actual, actual_loss = sdpa(self.tokens, self.tokens)
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)
                torch.testing.assert_close(actual_loss, expected_loss)
                actual_loss.backward()
                expected_loss.backward()
                for a, b in zip(sdpa.parameters(), manual.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=1e-4)

    def test_sdpa_masks_and_attention_weights(self):
        attention = ATTENTION.MultiHeadSelfAttention(16, 4, attention_backend='sdpa').eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad(), patch.object(
            ATTENTION.F, 'scaled_dot_product_attention', wraps=F.scaled_dot_product_attention,
        ) as sdpa:
            _, cache = attention(x[:, :3], use_cache=True)
            self.assertTrue(sdpa.call_args.kwargs['is_causal'])
            self.assertIsNone(sdpa.call_args.kwargs['attn_mask'])
            output = attention(x[:, 3:], past_key_value=cache)
            self.assertFalse(sdpa.call_args.kwargs['is_causal'])
            torch.testing.assert_close(
                sdpa.call_args.kwargs['attn_mask'],
                torch.tensor([[True, True, True, True, False], [True] * 5]),
            )
            sdpa.reset_mock()
            manual_output, weights, present = attention(
                x[:, 3:], past_key_value=cache, use_cache=True, return_attention=True,
            )
            sdpa.assert_not_called()
            torch.testing.assert_close(output, manual_output)
            self.assertEqual(weights.shape, (2, 4, 2, 5))
            self.assertEqual(weights[:, :, 0, 4].count_nonzero().item(), 0)
            torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 2))
            self.assertEqual(present[0].shape, (2, 4, 5, 4))

    def test_sdpa_cpu_fallback(self):
        model = self.model(attention_backend='sdpa').eval()
        with torch.no_grad():
            expected, _ = model(self.tokens)
            with patch.object(ATTENTION.F, 'scaled_dot_product_attention', side_effect=NotImplementedError):
                actual, _ = model(self.tokens)
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)
            with patch.object(ATTENTION.F, 'scaled_dot_product_attention'):
                del ATTENTION.F.scaled_dot_product_attention
                actual, _ = model(self.tokens)
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)

    def test_rope_known_rotation_and_no_learned_positions(self):
        model = self.model(position_encoding='rope')
        self.assertIsNone(model.position_embedding)
        self.assertNotIn('position_embedding.weight', model.state_dict())
        attention = ATTENTION.MultiHeadSelfAttention(2, 1, position_encoding='rope')
        x = torch.tensor([[[[1.0, 0.0]]]])
        torch.testing.assert_close(attention._apply_rope(x, 0), x)
        torch.testing.assert_close(
            attention._apply_rope(x, 3),
            torch.tensor([[[[torch.cos(torch.tensor(3.0)), torch.sin(torch.tensor(3.0))]]]]),
        )

    def test_cache_inference_only(self):
        model = self.model()
        with self.assertRaises(ValueError):
            model(self.tokens, use_cache=True)
        model.eval()
        with torch.no_grad():
            _, _, cache = model(self.tokens[:, :2], use_cache=True)
        for use_cache in (False, True):
            with self.assertRaises(ValueError):
                model(self.tokens[:, 2:3], self.tokens[:, 2:3], past_key_values=cache, use_cache=use_cache)
        with self.assertRaises(ValueError):
            model(self.tokens, self.tokens, use_cache=True)
        model.train()
        with self.assertRaises(ValueError):
            model(self.tokens[:, 2:3], past_key_values=cache)

    def test_invalid_cache(self):
        model = self.model().eval()
        with torch.no_grad():
            _, _, cache = model(self.tokens[:, :2], use_cache=True)
        k, v = cache[0]
        invalid = [
            (), cache[:1], (None, cache[1]), ((k,), cache[1]),
            ((k[0], v[0]), cache[1]), ((k[:1], v[:1]), cache[1]),
            ((k[:, :1], v[:, :1]), cache[1]), ((k[..., :2], v[..., :2]), cache[1]),
            ((k, v[:, :, :1]), cache[1]), ((k.double(), v), cache[1]),
            ((k.double(), v.double()), cache[1]), ((k.long(), v.long()), cache[1]),
            ((k[:, :, :1], v[:, :, :1]), cache[1]),
            ((k.to('meta'), v.to('meta')), cache[1]),
        ]
        for bad in invalid:
            with self.subTest(cache=repr(bad)[:60]), self.assertRaises(ValueError):
                model(self.tokens[:, :1], past_key_values=bad)

    def test_context_limit_and_empty_inputs(self):
        model = self.model().eval()
        with torch.no_grad():
            _, _, cache = model(self.tokens, use_cache=True)
        for x, past in (
            (self.tokens[:, :1], cache), (self.tokens.repeat(1, 2), None),
            (self.tokens[:, :0], None), (self.tokens[0], None),
        ):
            with self.assertRaises(ValueError):
                model(x, past_key_values=past)

    def test_autocast_cache_dtype(self):
        model = self.model(position_encoding='rope', attention_backend='sdpa').eval()
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            full, _ = model(self.tokens)
            first, _, cache = model(self.tokens[:, :3], use_cache=True)
            self.assertEqual(cache[0][0].dtype, torch.bfloat16)
            last, _, _ = model(self.tokens[:, 3:], past_key_values=cache, use_cache=True)
            torch.testing.assert_close(torch.cat((first, last), dim=1), full, atol=0.002, rtol=0.02)

    def test_invalid_configuration(self):
        for config in (
            {'position_encoding': 'bad'}, {'attention_backend': 'bad'},
            {'embedding_dim': 15}, {'embedding_dim': 12, 'position_encoding': 'rope'},
            {'dropout': -0.1}, {'dropout': 1.1}, {'dropout': float('nan')},
            *({name: value} for name in (
                'vocab_size', 'context_length', 'embedding_dim', 'num_heads', 'num_layers', 'ffn_dim',
            ) for value in (0, -1, 1.5, True)),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.model(**config)


if __name__ == '__main__':
    unittest.main()
