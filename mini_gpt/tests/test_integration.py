"""End-to-end checkpoint, training and generation regressions."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch


BASE_DIR = Path(__file__).resolve().parents[1]


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], BASE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_module = load_module('06_train.py')
generate_module = load_module('07_generate.py')


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_reload_character_defaults(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            checkpoint = train_module.train(
                checkpoint_dir=Path(directory), device='cpu', max_steps=1,
            )
            self.assertEqual(checkpoint['config']['embedding_dim'], 64)
            self.assertEqual(checkpoint['config']['context_length'], 32)
            self.assertEqual(checkpoint['config']['position_encoding'], 'learned')
            self.assertEqual(checkpoint['config']['attention_backend'], 'manual')
            self.assertEqual(len(checkpoint['metrics']['validation_loss']), 1)
            self.assertTrue((Path(directory) / 'loss_curve.png').exists())
            model, tokenizer = generate_module.load_model_and_tokenizer(
                Path(directory) / 'mini_gpt.pt', device='cpu',
            )
            self.assertEqual(tokenizer.itos, checkpoint['vocab'])
            output = generate_module.generate(
                model, tokenizer, 'bonjour ', 3, temperature=0, use_cache=True, verbose=False,
            )
            self.assertEqual(len(output), 11)

    def test_bpe_training_reload(self):
        try:
            import tokenizers  # noqa: F401
        except ImportError:
            self.skipTest('Optional tokenizers dependency not installed.')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(directory)
            corpus = directory / 'corpus.txt'
            corpus.write_text('bonjour le monde ! café et thé.\n' * 100, encoding='utf-8')
            checkpoint = train_module.train(
                corpus_path=corpus, checkpoint_dir=directory / 'run',
                tokenizer_kind='bpe', vocab_size=280, device='cpu',
                context_length=8, embedding_dim=16, ffn_dim=32,
                position_encoding='rope', attention_backend='sdpa', max_steps=1, stride=8,
            )
            model, tokenizer = generate_module.load_model_and_tokenizer(
                directory / 'run' / 'mini_gpt.pt', device='cpu',
            )
            original = train_module.tokenizer_module.tokenizer_from_state(checkpoint['tokenizer'])
            self.assertEqual(tokenizer.encode('café inconnu'), original.encode('café inconnu'))
            self.assertEqual(tokenizer.decode(tokenizer.encode('café')), 'café')
            for cache in (False, True):
                self.assertIsInstance(generate_module.generate(
                    model, tokenizer, 'bonjour', 12, temperature=0, use_cache=cache, verbose=False,
                ), str)

    def test_legacy_checkpoint(self):
        config = dict(train_module.MODEL_PRESETS['tiny'], dropout=0.1)
        tokenizer = generate_module.SimpleTokenizer.from_text('bonjour ')
        model = generate_module.MiniGPT(tokenizer.vocab_size, **config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            torch.save({
                'config': config, 'vocab': tokenizer.itos,
                'model_state_dict': model.state_dict(),
            }, path)
            loaded, restored = generate_module.load_model_and_tokenizer(path, device='cpu')
            self.assertEqual(restored.encode('bonjour'), tokenizer.encode('bonjour'))
            x = torch.tensor([tokenizer.encode('bonjour')])
            with torch.no_grad():
                torch.testing.assert_close(loaded(x)[0], model(x)[0])

    def test_cache_generation_across_window_boundary(self):
        tokenizer = generate_module.SimpleTokenizer.from_text('bonjour ')
        for position in ('learned', 'rope'):
            for backend in ('manual', 'sdpa'):
                model = generate_module.MiniGPT(
                    tokenizer.vocab_size, context_length=8, embedding_dim=16,
                    num_heads=2, num_layers=2, ffn_dim=32,
                    position_encoding=position, attention_backend=backend,
                )
                for prompt in ('bon', 'bonjour bonjour '):
                    with self.subTest(position=position, backend=backend, prompt=prompt):
                        outputs = [
                            generate_module.generate(
                                model, tokenizer, prompt, 12, temperature=0,
                                use_cache=cache, verbose=False,
                            )
                            for cache in (False, True)
                        ]
                        self.assertEqual(outputs[0], outputs[1])

    def test_empty_prompt_and_zero_generation(self):
        tokenizer = generate_module.SimpleTokenizer.from_text('abc')
        model = generate_module.MiniGPT(tokenizer.vocab_size, context_length=4)
        with self.assertRaises(ValueError):
            generate_module.generate(model, tokenizer, '', verbose=False)
        self.assertEqual(generate_module.generate(
            model, tokenizer, 'abc', max_new_tokens=0, verbose=False,
        ), 'abc')

    def test_split_precedes_windows(self):
        train_loader, val_loader = train_module.create_dataloaders(list(range(100)), 4, 2)
        self.assertEqual(train_loader.dataset.token_ids, list(range(90)))
        self.assertEqual(val_loader.dataset.token_ids, list(range(90, 100)))
        with self.assertRaises(ValueError):
            train_module.create_dataloaders(list(range(20)), 4, 2)

    def test_invalid_options(self):
        for options in ({'epochs': 0}, {'stride': 0}, {'max_steps': -1}, {'preset': 'unknown'},
                        {'precision': 'float16'}, {'learning_rate': float('nan')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                train_module.train(**options)

    def test_notebook_code_syntax(self):
        notebook = json.loads((BASE_DIR / 'mini_gpt_pedagogique.ipynb').read_text())
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'notebook-cell-{i}', 'exec')


if __name__ == '__main__':
    unittest.main()
