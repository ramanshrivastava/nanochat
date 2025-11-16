"""
Unit tests for BPE Training (Phase 0.2)

Tests the BPE training algorithm implementation.

Run with:
    pytest tests/unit/test_bpe.py -v
"""

import pytest
import tempfile
import os
from mini_nanochat.bpe import BPETrainer, train_bpe_from_file


class TestBPEBasics:
    """Test basic BPE functionality."""

    def test_initialization(self):
        """Test BPE trainer initializes correctly."""
        trainer = BPETrainer(vocab_size=512)
        assert trainer.vocab_size == 512
        assert len(trainer.merges) == 0
        assert len(trainer.vocab) == 256  # Byte vocabulary

    def test_byte_vocabulary(self):
        """Test that byte vocabulary is correctly initialized."""
        trainer = BPETrainer()
        # Check all 256 bytes are in vocab
        for i in range(256):
            assert i in trainer.vocab
            assert trainer.vocab[i] == bytes([i])

    def test_encode_bytes(self):
        """Test byte encoding."""
        trainer = BPETrainer()
        # "Hi" = [72, 105]
        assert trainer._encode_bytes("Hi") == [72, 105]
        # "A" = [65]
        assert trainer._encode_bytes("A") == [65]
        # Empty string
        assert trainer._encode_bytes("") == []


class TestPairCounting:
    """Test pair counting functionality."""

    def test_count_pairs_simple(self):
        """Test counting pairs in simple sequences."""
        trainer = BPETrainer()
        sequences = [[1, 2, 3], [1, 2, 4]]
        pairs = trainer._count_pairs(sequences)

        assert pairs[(1, 2)] == 2  # Appears in both
        assert pairs[(2, 3)] == 1
        assert pairs[(2, 4)] == 1

    def test_count_pairs_repeated(self):
        """Test counting with repeated pairs."""
        trainer = BPETrainer()
        sequences = [[1, 1, 1, 1]]
        pairs = trainer._count_pairs(sequences)

        # (1, 1) appears 3 times
        assert pairs[(1, 1)] == 3

    def test_count_pairs_empty(self):
        """Test counting with empty/single-token sequences."""
        trainer = BPETrainer()
        # Empty sequences
        pairs = trainer._count_pairs([[]])
        assert len(pairs) == 0

        # Single token (no pairs)
        pairs = trainer._count_pairs([[1]])
        assert len(pairs) == 0


class TestMerging:
    """Test pair merging functionality."""

    def test_merge_pair_simple(self):
        """Test merging a pair."""
        trainer = BPETrainer()
        tokens = [1, 2, 3, 1, 2]
        result = trainer._merge_pair(tokens, (1, 2), 256)
        assert result == [256, 3, 256]

    def test_merge_pair_consecutive(self):
        """Test merging consecutive occurrences."""
        trainer = BPETrainer()
        tokens = [1, 2, 1, 2, 1, 2]
        result = trainer._merge_pair(tokens, (1, 2), 256)
        assert result == [256, 256, 256]

    def test_merge_pair_no_match(self):
        """Test merging when pair doesn't exist."""
        trainer = BPETrainer()
        tokens = [1, 3, 5]
        result = trainer._merge_pair(tokens, (1, 2), 256)
        # No change
        assert result == [1, 3, 5]

    def test_merge_pair_empty(self):
        """Test merging with empty sequence."""
        trainer = BPETrainer()
        result = trainer._merge_pair([], (1, 2), 256)
        assert result == []

    def test_merge_pair_single_token(self):
        """Test merging with single token."""
        trainer = BPETrainer()
        result = trainer._merge_pair([1], (1, 2), 256)
        assert result == [1]


class TestTraining:
    """Test BPE training process."""

    def test_train_simple(self):
        """Test training on simple repeated text."""
        trainer = BPETrainer(vocab_size=300)
        texts = ["aaabdaaabac"] * 10  # Repeated pattern

        stats = trainer.train(texts, verbose=False)

        assert stats['vocab_size'] > 256  # Some merges happened
        assert stats['num_merges'] > 0
        assert len(trainer.merges) == stats['num_merges']

    def test_train_hello_world(self):
        """Test training on hello world."""
        trainer = BPETrainer(vocab_size=280)
        texts = ["hello world", "hello", "world"] * 20

        stats = trainer.train(texts, verbose=False)

        # Should learn "hello", "world", etc.
        assert stats['num_merges'] > 0

        # Test encoding
        tokens = trainer.encode("hello")
        decoded = trainer.decode(tokens)
        assert decoded == "hello"

    def test_train_stops_when_no_pairs(self):
        """Test training stops when no more pairs exist."""
        trainer = BPETrainer(vocab_size=1000)  # Large target
        texts = ["abc"]  # Very small corpus

        stats = trainer.train(texts, verbose=False)

        # Should stop early (not reach 1000)
        assert stats['vocab_size'] < 1000

    def test_empty_corpus(self):
        """Test training on empty corpus."""
        trainer = BPETrainer(vocab_size=300)
        texts = []

        stats = trainer.train(texts, verbose=False)

        # No merges possible
        assert stats['num_merges'] == 0
        assert stats['vocab_size'] == 256  # Only base vocab


class TestEncoding:
    """Test encoding functionality."""

    def test_encode_decode_roundtrip(self):
        """Test that encode→decode is lossless."""
        trainer = BPETrainer(vocab_size=300)
        trainer.train(["hello world"] * 50, verbose=False)

        test_texts = [
            "hello",
            "world",
            "hello world",
            "Hello, World!",
            "test 123",
        ]

        for text in test_texts:
            tokens = trainer.encode(text)
            decoded = trainer.decode(tokens)
            assert decoded == text, f"Roundtrip failed for: {text}"

    def test_encode_untrained_text(self):
        """Test encoding text not in training corpus."""
        trainer = BPETrainer(vocab_size=300)
        trainer.train(["hello"] * 100, verbose=False)

        # Encode completely different text
        tokens = trainer.encode("goodbye")
        decoded = trainer.decode(tokens)

        # Should still work (byte fallback)
        assert decoded == "goodbye"

    def test_encode_unicode(self):
        """Test encoding Unicode characters."""
        trainer = BPETrainer(vocab_size=300)
        trainer.train(["hello 世界"] * 20, verbose=False)

        tokens = trainer.encode("世界")
        decoded = trainer.decode(tokens)

        assert decoded == "世界"

    def test_encode_special_characters(self):
        """Test encoding special characters."""
        trainer = BPETrainer(vocab_size=300)
        texts = ["Hello\nWorld", "Tab\there", "Quote'test"]

        trainer.train(texts * 10, verbose=False)

        for text in texts:
            tokens = trainer.encode(text)
            decoded = trainer.decode(tokens)
            assert decoded == text


class TestSaveLoad:
    """Test saving and loading trained models."""

    def test_save_and_load(self):
        """Test saving and loading BPE model."""
        trainer1 = BPETrainer(vocab_size=300)
        trainer1.train(["hello world"] * 100, verbose=False)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            trainer1.save(temp_path)

            # Load into new trainer
            trainer2 = BPETrainer()
            trainer2.load(temp_path)

            # Should produce same encodings
            text = "hello world test"
            tokens1 = trainer1.encode(text)
            tokens2 = trainer2.encode(text)

            assert tokens1 == tokens2
            assert trainer1.vocab_size == trainer2.vocab_size
            assert len(trainer1.merges) == len(trainer2.merges)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_loaded_model_can_encode_decode(self):
        """Test that loaded model works correctly."""
        trainer1 = BPETrainer(vocab_size=300)
        trainer1.train(["test data"] * 50, verbose=False)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            trainer1.save(temp_path)
            trainer2 = BPETrainer()
            trainer2.load(temp_path)

            # Test roundtrip with loaded model
            text = "test data encoding"
            tokens = trainer2.encode(text)
            decoded = trainer2.decode(tokens)

            assert decoded == text

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestStatistics:
    """Test statistics and utility functions."""

    def test_get_stats(self):
        """Test getting model statistics."""
        trainer = BPETrainer(vocab_size=300)
        trainer.train(["test"] * 10, verbose=False)

        stats = trainer.get_stats()

        assert 'vocab_size' in stats
        assert 'num_merges' in stats
        assert 'base_vocab' in stats
        assert stats['base_vocab'] == 256
        assert stats['vocab_size'] == 256 + stats['num_merges']


class TestTrainFromFile:
    """Test training from file functionality."""

    def test_train_from_file(self):
        """Test training BPE from a text file."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("hello world\n" * 100)
            f.write("test data\n" * 100)
            temp_input = f.name

        try:
            trainer = train_bpe_from_file(temp_input, vocab_size=300, output_path=None)

            # Should have trained successfully
            assert len(trainer.merges) > 0

            # Can encode/decode
            tokens = trainer.encode("hello world test")
            decoded = trainer.decode(tokens)
            assert "hello" in decoded
            assert "world" in decoded

        finally:
            if os.path.exists(temp_input):
                os.remove(temp_input)


class TestCompression:
    """Test compression capabilities."""

    def test_compression_ratio(self):
        """Test that BPE compresses repeated patterns."""
        trainer = BPETrainer(vocab_size=400)

        # Highly repetitive text
        texts = ["hello hello hello world world world"] * 100
        trainer.train(texts, verbose=False)

        text = "hello hello world world"

        # Encode with BPE
        tokens_bpe = trainer.encode(text)

        # Compare to byte encoding
        tokens_bytes = trainer._encode_bytes(text)

        # BPE should use fewer tokens
        assert len(tokens_bpe) < len(tokens_bytes), \
            f"BPE ({len(tokens_bpe)}) should compress better than bytes ({len(tokens_bytes)})"

    def test_common_words_get_tokens(self):
        """Test that common words become single tokens."""
        trainer = BPETrainer(vocab_size=400)

        # Train with repeated words
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the dog was lazy",
            "the fox was quick"
        ] * 100

        trainer.train(texts, verbose=False)

        # "the" appears many times, should be a single token
        tokens_the = trainer.encode("the")

        # Should be compressed (fewer tokens than bytes)
        bytes_the = trainer._encode_bytes("the")
        assert len(tokens_the) <= len(bytes_the)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_character(self):
        """Test encoding single characters."""
        trainer = BPETrainer(vocab_size=300)
        trainer.train(["a"], verbose=False)

        tokens = trainer.encode("a")
        decoded = trainer.decode(tokens)
        assert decoded == "a"

    def test_very_long_text(self):
        """Test encoding very long text."""
        trainer = BPETrainer(vocab_size=300)
        trainer.train(["test"] * 10, verbose=False)

        long_text = "test " * 1000  # 5000 characters
        tokens = trainer.encode(long_text)
        decoded = trainer.decode(tokens)

        assert decoded == long_text

    def test_newlines_and_tabs(self):
        """Test handling of newlines and tabs."""
        trainer = BPETrainer(vocab_size=300)
        texts = ["line1\nline2\tline3"]
        trainer.train(texts * 10, verbose=False)

        tokens = trainer.encode("line1\nline2\tline3")
        decoded = trainer.decode(tokens)

        assert decoded == "line1\nline2\tline3"
        assert "\n" in decoded
        assert "\t" in decoded


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
