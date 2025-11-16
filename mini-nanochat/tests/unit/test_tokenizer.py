"""
Unit tests for Tokenizer (Phase 0.1)

Tests the basic tokenizer wrapper functionality using tiktoken/GPT-2.

Run with:
    pytest tests/unit/test_tokenizer.py -v
"""

import pytest
from mini_nanochat.tokenizer import Tokenizer


class TestTokenizerBasics:
    """Test basic tokenizer functionality."""

    def test_initialization(self):
        """Test tokenizer initializes correctly."""
        tok = Tokenizer()
        assert tok is not None
        assert tok.vocab_size == 50257  # GPT-2 vocab size

    def test_vocab_size(self):
        """Test vocabulary size is correct."""
        tok = Tokenizer()
        assert len(tok) == 50257

    def test_repr(self):
        """Test string representation."""
        tok = Tokenizer()
        repr_str = repr(tok)
        assert "Tokenizer" in repr_str
        assert "50257" in repr_str


class TestEncoding:
    """Test encoding text to tokens."""

    def test_encode_simple(self):
        """Test encoding simple text."""
        tok = Tokenizer()
        tokens = tok.encode("Hello")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_encode_hello_world(self):
        """Test encoding 'Hello, world!'."""
        tok = Tokenizer()
        tokens = tok.encode("Hello, world!")

        # Should produce 4 tokens for GPT-2:
        # "Hello", ",", " world", "!"
        assert len(tokens) == 4
        assert tokens == [15496, 11, 995, 0]

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        tok = Tokenizer()
        tokens = tok.encode("")
        assert tokens == []

    def test_encode_whitespace(self):
        """Test that whitespace matters."""
        tok = Tokenizer()

        # With leading space
        tokens_with_space = tok.encode(" hello")

        # Without leading space
        tokens_no_space = tok.encode("hello")

        # Should be different
        assert tokens_with_space != tokens_no_space

    def test_encode_numbers(self):
        """Test encoding numbers."""
        tok = Tokenizer()

        # Small number (likely single token)
        tokens_2024 = tok.encode("2024")

        # Should tokenize
        assert len(tokens_2024) >= 1

    def test_encode_special_characters(self):
        """Test encoding special characters."""
        tok = Tokenizer()

        special_texts = [
            "Hello\nWorld",  # Newline
            "Hello\tWorld",  # Tab
            "Hello 😊",  # Emoji
            "Hello 世界",  # Chinese
        ]

        for text in special_texts:
            tokens = tok.encode(text)
            # Should encode without error
            assert len(tokens) > 0


class TestDecoding:
    """Test decoding tokens to text."""

    def test_decode_simple(self):
        """Test decoding simple token sequence."""
        tok = Tokenizer()
        tokens = [15496]  # "Hello"
        text = tok.decode(tokens)
        assert text == "Hello"

    def test_decode_hello_world(self):
        """Test decoding 'Hello, world!'."""
        tok = Tokenizer()
        tokens = [15496, 11, 995, 0]  # "Hello, world!"
        text = tok.decode(tokens)
        assert text == "Hello, world!"

    def test_decode_empty(self):
        """Test decoding empty token list."""
        tok = Tokenizer()
        text = tok.decode([])
        assert text == ""

    def test_encode_decode_roundtrip(self):
        """Test that encode→decode is lossless."""
        tok = Tokenizer()

        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "The year is 2024.",
            "123 456 789",
            "Special: @#$%",
            "Newline\nTab\t",
        ]

        for original_text in test_texts:
            tokens = tok.encode(original_text)
            decoded_text = tok.decode(tokens)
            assert decoded_text == original_text, f"Roundtrip failed for: {original_text}"


class TestBatchProcessing:
    """Test batch encoding/decoding."""

    def test_encode_batch(self):
        """Test batch encoding."""
        tok = Tokenizer()
        texts = ["Hello", "World", "Test"]
        batch_tokens = tok.encode_batch(texts)

        assert len(batch_tokens) == 3
        assert all(isinstance(tokens, list) for tokens in batch_tokens)

    def test_decode_batch(self):
        """Test batch decoding."""
        tok = Tokenizer()
        token_lists = [[15496], [10603], [14402]]  # "Hello", "World", "Test"
        texts = tok.decode_batch(token_lists)

        assert len(texts) == 3
        assert texts[0] == "Hello"
        assert texts[1] == "World"
        assert texts[2] == "Test"

    def test_batch_roundtrip(self):
        """Test batch encode→decode roundtrip."""
        tok = Tokenizer()
        original_texts = ["Hello", "World", "Test", "Batch"]

        tokens = tok.encode_batch(original_texts)
        decoded_texts = tok.decode_batch(tokens)

        assert decoded_texts == original_texts


class TestUtilityMethods:
    """Test utility analysis methods."""

    def test_tokenize_and_count(self):
        """Test tokenize_and_count method."""
        tok = Tokenizer()
        stats = tok.tokenize_and_count("Hello, world!")

        assert "tokens" in stats
        assert "num_tokens" in stats
        assert "num_chars" in stats
        assert "compression_ratio" in stats

        # "Hello, world!" = 13 chars, 4 tokens
        assert stats["num_chars"] == 13
        assert stats["num_tokens"] == 4
        assert stats["compression_ratio"] == 13 / 4

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        tok = Tokenizer()

        # English text should have good compression (4-5 chars/token)
        text = "The quick brown fox jumps over the lazy dog."
        stats = tok.tokenize_and_count(text)

        # Should be reasonable compression
        assert 3.0 <= stats["compression_ratio"] <= 6.0

    def test_analyze_tokenization(self):
        """Test detailed tokenization analysis."""
        tok = Tokenizer()

        # Test with verbose=False (no printing)
        result = tok.analyze_tokenization("Hello", verbose=False)

        assert "text" in result
        assert "tokens" in result
        assert "token_strings" in result
        assert "stats" in result

        assert result["text"] == "Hello"
        assert len(result["tokens"]) == len(result["token_strings"])


class TestCompressionAnalysis:
    """Test compression ratio for different text types."""

    def test_english_compression(self):
        """Test compression ratio for English text."""
        tok = Tokenizer()

        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test of tokenization compression."
        )
        stats = tok.tokenize_and_count(text)

        # English prose: should be ~4-5 chars/token
        assert 3.5 <= stats["compression_ratio"] <= 6.0

    def test_code_compression(self):
        """Test compression ratio for code."""
        tok = Tokenizer()

        code = """def hello():
    print('Hello, world!')
    return True"""

        stats = tok.tokenize_and_count(code)

        # Code: should be ~3-4 chars/token (more symbols)
        assert 2.0 <= stats["compression_ratio"] <= 5.0

    def test_number_compression(self):
        """Test compression ratio for numbers."""
        tok = Tokenizer()

        numbers = "123 456 789 1234 5678 9012"
        stats = tok.tokenize_and_count(numbers)

        # Numbers: variable compression depending on vocabulary
        assert stats["compression_ratio"] > 0


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_very_long_text(self):
        """Test with very long text."""
        tok = Tokenizer()

        # 10,000 character text
        long_text = "Hello world! " * 1000
        tokens = tok.encode(long_text)

        # Should handle without error
        assert len(tokens) > 0

        # Roundtrip should work
        decoded = tok.decode(tokens)
        assert decoded == long_text

    def test_unicode_characters(self):
        """Test with various Unicode characters."""
        tok = Tokenizer()

        unicode_texts = [
            "Hello 世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "Hello 🌍🌎🌏",  # Emojis
        ]

        for text in unicode_texts:
            tokens = tok.encode(text)
            decoded = tok.decode(tokens)
            assert decoded == text

    def test_only_whitespace(self):
        """Test with only whitespace."""
        tok = Tokenizer()

        whitespace_texts = [
            " ",
            "  ",
            "\n",
            "\t",
            "   \n\t  ",
        ]

        for text in whitespace_texts:
            tokens = tok.encode(text)
            decoded = tok.decode(tokens)
            assert decoded == text


class TestConsistency:
    """Test consistency with expected behavior."""

    def test_deterministic(self):
        """Test that encoding is deterministic."""
        tok = Tokenizer()

        text = "Hello, world!"

        # Encode multiple times
        tokens1 = tok.encode(text)
        tokens2 = tok.encode(text)
        tokens3 = tok.encode(text)

        # Should always produce same result
        assert tokens1 == tokens2 == tokens3

    def test_whitespace_preservation(self):
        """Test that whitespace is preserved exactly."""
        tok = Tokenizer()

        texts_with_whitespace = [
            "Hello world",
            "Hello  world",  # Two spaces
            "Hello   world",  # Three spaces
            "Hello\nworld",  # Newline
            " Hello",  # Leading space
            "Hello ",  # Trailing space
        ]

        for text in texts_with_whitespace:
            tokens = tok.encode(text)
            decoded = tok.decode(tokens)
            assert decoded == text, f"Whitespace not preserved in: {repr(text)}"


# Parametrized tests for various inputs
@pytest.mark.parametrize(
    "text,expected_tokens",
    [
        ("Hello", [15496]),
        ("World", [10603]),
        ("!", [0]),
        (" ", [220]),
    ],
)
def test_specific_tokens(text, expected_tokens):
    """Test specific text produces expected tokens."""
    tok = Tokenizer()
    tokens = tok.encode(text)
    assert tokens == expected_tokens


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "The quick brown fox",
        "123 456 789",
        "Special: @#$%^&*()",
        "Unicode: 世界 🌍",
    ],
)
def test_roundtrip_parametrized(text):
    """Parametrized test for encode→decode roundtrip."""
    tok = Tokenizer()
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    assert decoded == text


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
