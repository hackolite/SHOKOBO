import builtins
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('test_dataset_module', BASE_DIR / '02_dataset.py')
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)
tokenizers = data.tokenizer_module
HAS_BPE = importlib.util.find_spec('tokenizers') is not None


class TokenizerTests(unittest.TestCase):
    def test_legacy_char_and_state(self):
        tokenizer = tokenizers.SimpleTokenizer.from_text('é🙂aba')
        self.assertEqual(tokenizer.decode(tokenizer.encode('a🙂é')), 'a🙂é')
        self.assertEqual(tokenizer.decode(tokenizer.encode('z')), '?')
        restored = tokenizers.tokenizer_from_state(json.loads(json.dumps(tokenizer.to_state())))
        self.assertEqual(restored.itos, tokenizer.itos)
        self.assertEqual(restored.encode('a🙂z'), tokenizer.encode('a🙂z'))

    def test_stream_char(self):
        tokenizer = tokenizers.train_tokenizer(iter(['ba', '', 'éa']))
        self.assertEqual(tokenizer.itos, ['<unk>', 'a', 'b', 'é'])

    def test_empty_and_unknown_kind(self):
        for kind in ('char', 'bpe'):
            with self.subTest(kind=kind):
                if kind == 'bpe' and not HAS_BPE:
                    continue
                with self.assertRaisesRegex(ValueError, 'empty'):
                    tokenizers.train_tokenizer(iter(['', '']), kind=kind)
        with self.assertRaises(ValueError):
            tokenizers.train_tokenizer(['abc'], kind='invalid')
        with self.assertRaises(ValueError):
            tokenizers.tokenizer_from_state({'kind': 'invalid'})

    def test_bpe_is_optional(self):
        original_import = builtins.__import__

        def without_bpe(name, *args, **kwargs):
            if name == 'tokenizers':
                raise ImportError('not installed')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=without_bpe):
            self.assertEqual(tokenizers.train_tokenizer(['abc']).vocab_size, 4)
            with self.assertRaisesRegex(ImportError, 'requirements-bpe.txt'):
                tokenizers.train_tokenizer(['abc'], kind='bpe')

    @unittest.skipUnless(HAS_BPE, 'optional tokenizers package not installed')
    def test_bpe_roundtrip_state_and_unseen_unicode(self):
        tokenizer = tokenizers.train_tokenizer(['hello world!\n'] * 20, 'bpe', 280)
        text = 'hello\r\n🙂你好 é <unk> world'
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        restored = tokenizers.tokenizer_from_state(json.loads(json.dumps(tokenizer.to_state())))
        self.assertEqual(tokenizer.vocab_size, restored.vocab_size)
        self.assertEqual(tokenizer.encode(text), restored.encode(text))
        self.assertEqual(restored.decode(restored.encode(text)), text)
        self.assertEqual(json.loads(tokenizer.to_state()['tokenizer_json'])['model']['type'], 'BPE')

    def test_bpe_training_budget(self):
        with patch.object(tokenizers, 'BPE_TRAINING_CHAR_LIMIT', 7), patch.object(
            tokenizers, 'BPE_CHUNK_SIZE', 3
        ):
            self.assertEqual(
                list(tokenizers._bpe_training_chunks(iter(['', 'abcd', 'efghijk', 'unused']))),
                ['abc', 'd', 'efg'],
            )


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.root = BASE_DIR / 'tests' / ('.test-data-' + uuid.uuid4().hex)
        self.root.mkdir()
        self.corpus = self.root / 'corpus.txt'
        self.output = self.root / 'prepared'

    def tearDown(self):
        shutil.rmtree(self.root)

    def write_text(self, text):
        with self.corpus.open('w', encoding='utf-8', newline='') as target:
            target.write(text)

    def test_disjoint_raw_split_and_train_only_vocabulary(self):
        self.write_text('abcabcZZ')
        tokenizer, train, val = data.prepare_corpus(self.corpus, self.output, validation_fraction=0.25)
        self.assertNotIn('Z', tokenizer.stoi)
        self.assertEqual(np.fromfile(train, dtype=np.int64).tolist(), tokenizer.encode('abcabc'))
        self.assertEqual(np.fromfile(val, dtype=np.int64).tolist(), [tokenizer.unk_id] * 2)
        state = json.loads((self.output / 'tokenizer.json').read_text(encoding='utf-8'))
        restored = data.tokenizer_from_state(state)
        self.assertEqual(restored.itos, tokenizer.itos)

    def test_unicode_crlf_and_giant_line_bounded_reads(self):
        with self.corpus.open('w', encoding='utf-8', newline='') as target:
            for _ in range(1000):
                target.write('é🙂abc' * 100)
            target.write('\r\né🙂')
        original_open = Path.open
        read_sizes = []

        class CheckedReader:
            def __init__(self, file):
                self.file = file

            def __enter__(self):
                self.file.__enter__()
                return self

            def __exit__(self, *args):
                return self.file.__exit__(*args)

            def read(self, size=-1):
                self.assert_size(size)
                read_sizes.append(size)
                return self.file.read(size)

            def assert_size(self, size):
                if not 0 < size <= data.TEXT_CHUNK_SIZE:
                    raise AssertionError(f'Unbounded read: {size}')

        def checked_open(path, *args, **kwargs):
            file = original_open(path, *args, **kwargs)
            return CheckedReader(file) if path == self.corpus else file

        with patch.object(Path, 'open', checked_open):
            tokenizer, train, val = data.prepare_corpus(self.corpus, self.output, validation_fraction=0.2)
        self.assertGreater(len(read_sizes), 100)
        total = 500_004
        self.assertEqual(train.stat().st_size // 8, int(total * 0.8))
        self.assertEqual(val.stat().st_size // 8, total - int(total * 0.8))
        # CRLF was never normalized; both characters are unseen in training.
        self.assertEqual(np.fromfile(val, dtype=np.int64)[-4:].tolist(), tokenizer.encode('\r\né🙂'))

    def test_invalid_and_short_splits(self):
        for text in ('', 'a', 'abc'):
            self.write_text(text)
            with self.assertRaisesRegex(ValueError, 'at least two characters'):
                data.prepare_corpus(self.corpus, self.output, validation_fraction=0.5)
        self.write_text('abcdef')
        for fraction in (0, 1, -0.1, float('nan'), float('inf')):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, 'validation_fraction'):
                data.prepare_corpus(self.corpus, self.output, validation_fraction=fraction)

    def test_short_encoded_split_preserves_previous_artifacts(self):
        self.write_text('abcdef')
        self.output.mkdir()
        (self.output / 'train.bin').write_bytes(b'previous')

        class CompressEverything:
            def encode(self, text):
                return [1]

        with patch.object(data, 'train_tokenizer', return_value=CompressEverything()):
            with self.assertRaisesRegex(ValueError, 'at least two'):
                data.prepare_corpus(self.corpus, self.output, validation_fraction=0.5)
        self.assertEqual((self.output / 'train.bin').read_bytes(), b'previous')
        self.assertFalse(list(self.output.glob('*.partial')))

    @unittest.skipUnless(HAS_BPE, 'optional tokenizers package not installed')
    def test_bpe_split_roundtrip_and_no_validation_training(self):
        train_text = 'hello hello world! ' * 20
        val_text = '🦊validation only' * 20
        self.write_text(train_text + val_text)
        fraction = len(val_text) / (len(train_text) + len(val_text))
        with patch.object(data, 'TEXT_CHUNK_SIZE', 31):
            tokenizer, train, val = data.prepare_corpus(
                self.corpus, self.output, 'bpe', 280, validation_fraction=fraction,
            )
            split = int((len(train_text) + len(val_text)) * (1 - fraction))
            raw = train_text + val_text
            expected = tokenizers.train_tokenizer(
                data._text_chunks(self.corpus, stop=split), 'bpe', 280,
            )
        self.assertEqual(tokenizer.backend.get_vocab(), expected.backend.get_vocab())
        self.assertEqual(tokenizer.decode(np.fromfile(train, dtype=np.int64).tolist()), raw[:split])
        self.assertEqual(tokenizer.decode(np.fromfile(val, dtype=np.int64).tolist()), raw[split:])

    def test_memmap_windows_stride_bounds_and_pickling(self):
        path = self.root / 'ids.bin'
        np.arange(11, dtype=np.int64).tofile(path)
        dataset = data.MemmapTokenDataset(path, context_length=4, stride=3)
        self.assertIsNone(dataset._tokens)
        self.assertEqual(len(dataset), 3)
        x, y = dataset[-1]
        self.assertEqual(x.tolist(), [6, 7, 8, 9])
        self.assertEqual(y.tolist(), [7, 8, 9, 10])
        self.assertEqual(x.dtype, torch.long)
        x[0] = 999
        self.assertEqual(dataset[-1][0][0].item(), 6)
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(restored._tokens)
        self.assertEqual(restored[0][0].tolist(), [0, 1, 2, 3])
        for index in (3, -4):
            with self.assertRaises(IndexError):
                dataset[index]
        with self.assertRaises(TypeError):
            dataset[1.5]
        self.assertFalse(any(isinstance(value, list) for value in dataset.__dict__.values()))

    def test_memmap_matches_legacy_and_batches(self):
        path = self.root / 'ids.bin'
        values = list(range(20))
        np.asarray(values, dtype=np.int64).tofile(path)
        mapped = data.MemmapTokenDataset(path, 4)
        legacy = data.NextTokenDataset(values, 4)
        self.assertEqual(len(mapped), len(legacy))
        for index in range(len(legacy)):
            for actual, expected in zip(mapped[index], legacy[index]):
                self.assertTrue(torch.equal(actual, expected))
        x, y = next(iter(torch.utils.data.DataLoader(mapped, batch_size=3)))
        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_invalid_memmap(self):
        path = self.root / 'ids.bin'
        np.arange(5, dtype=np.int64).tofile(path)
        for context, stride in ((0, 1), (2, 0), (2, -1), (2.5, 1), (True, 1), (2, 1.5)):
            with self.subTest(context=context, stride=stride), self.assertRaises(ValueError):
                data.MemmapTokenDataset(path, context, stride)
        with self.assertRaisesRegex(ValueError, 'context_length'):
            data.MemmapTokenDataset(path, 5)
        path.write_bytes(b'')
        with self.assertRaises(ValueError):
            data.MemmapTokenDataset(path, 1)
        path.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'eight bytes'):
            data.MemmapTokenDataset(path, 1)


if __name__ == '__main__':
    unittest.main()
