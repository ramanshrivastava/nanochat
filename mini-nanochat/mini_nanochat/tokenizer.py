"""
Tokenizer for mini-nanochat

Phase 0.1: Basic wrapper around HuggingFace tokenizers (GPT-2 style)

Historical Context:
- Uses byte-level BPE (GPT-2 style, 2019)
- ~50K vocabulary
- No UNK token (can encode any UTF-8 string)

Future: Phase 0.2 will implement custom BPE training

References:
- nanochat/tokenizer.py:1-100
- GPT-2 paper (2019): Section 2.2
- HuggingFace tokenizers: https://github.com/huggingface/tokenizers
"""

from tokenizers import Tokenizer as HFTokenizer
from typing import List


class Tokenizer:
    """
    Simple tokenizer wrapper using HuggingFace tokenizers.

    Phase 0.1: Uses GPT-2 pretrained tokenizer (~50K vocab)
    Phase 0.2: Will support custom vocabularies

    Design Decision (ADR-001):
    - Start with existing tokenizer (GPT-2) for simplicity
    - Implement custom BPE training later (Phase 0.2)
    - Use HuggingFace tokenizers (same as nanochat)

    Example:
        >>> tok = Tokenizer()
        >>> tokens = tok.encode("Hello, world!")
        >>> text = tok.decode(tokens)
        >>> print(text)
        Hello, world!
    """

    def __init__(self, vocab_size: int = None):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Vocabulary size (optional, inferred from model)

        Note:
            Phase 0.1: Uses GPT-2 pretrained model from HuggingFace
            Phase 0.2: Will support custom trained models
        """
        # Load GPT-2 pretrained tokenizer from HuggingFace
        # This downloads from HuggingFace hub
        try:
            self.tokenizer = HFTokenizer.from_pretrained("gpt2")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load GPT-2 tokenizer from HuggingFace: {e}\n"
                "Make sure you have internet connection and 'tokenizers' installed:\n"
                "  pip install tokenizers"
            )

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text to tokenize

        Returns:
            List of token IDs

        Example:
            >>> tok = Tokenizer()
            >>> tok.encode("Hello, world!")
            [15496, 11, 995, 0]

        Implementation Notes:
            - Uses byte-level BPE (no UNK token)
            - Can encode ANY UTF-8 string
            - Whitespace matters: "hello" ≠ " hello"

        Comparison to nanochat:
            - nanochat uses custom 65K vocab
            - nanochat has 8 special tokens
            - We use GPT-2 (~50K vocab)
        """
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string

        Example:
            >>> tok = Tokenizer()
            >>> tok.decode([15496, 11, 995, 0])
            'Hello, world!'

        Implementation Notes:
            - Handles byte-level decoding
            - Preserves whitespace exactly
            - Invalid token IDs will raise error
        """
        return self.tokenizer.decode(tokens)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts (batch processing).

        Args:
            texts: List of strings to encode

        Returns:
            List of token ID lists

        Example:
            >>> tok = Tokenizer()
            >>> tok.encode_batch(["Hello", "World"])
            [[15496], [10603]]
        """
        encodings = self.tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """
        Decode multiple token sequences (batch processing).

        Args:
            token_lists: List of token ID lists

        Returns:
            List of decoded strings

        Example:
            >>> tok = Tokenizer()
            >>> tok.decode_batch([[15496], [10603]])
            ['Hello', 'World']
        """
        return self.tokenizer.decode_batch(token_lists)

    def __len__(self) -> int:
        """
        Get vocabulary size.

        Returns:
            Number of tokens in vocabulary

        Example:
            >>> tok = Tokenizer()
            >>> len(tok)
            50257
        """
        return self.vocab_size

    def __repr__(self) -> str:
        """String representation of tokenizer."""
        return f"Tokenizer(vocab_size={self.vocab_size}, model='gpt2')"

    # Utility methods for analysis

    def tokenize_and_count(self, text: str) -> dict:
        """
        Tokenize text and return statistics.

        Args:
            text: Input text

        Returns:
            Dictionary with:
            - tokens: List of token IDs
            - num_tokens: Number of tokens
            - num_chars: Number of characters
            - compression_ratio: chars per token

        Example:
            >>> tok = Tokenizer()
            >>> stats = tok.tokenize_and_count("Hello, world!")
            >>> print(stats['compression_ratio'])
            3.25  # "Hello, world!" = 13 chars, 4 tokens

        Learning Note:
            Compression ratio is a key metric for tokenizers.
            - English prose: ~4-5 chars/token
            - Code: ~3-4 chars/token
            - JSON/data: ~2-3 chars/token
        """
        tokens = self.encode(text)
        num_chars = len(text)
        num_tokens = len(tokens)

        return {
            "tokens": tokens,
            "num_tokens": num_tokens,
            "num_chars": num_chars,
            "compression_ratio": num_chars / num_tokens if num_tokens > 0 else 0,
        }

    def analyze_tokenization(self, text: str, verbose: bool = True) -> dict:
        """
        Detailed analysis of how text is tokenized.

        Args:
            text: Input text
            verbose: If True, print detailed breakdown

        Returns:
            Dictionary with tokenization details

        Example:
            >>> tok = Tokenizer()
            >>> tok.analyze_tokenization("Hello, world!")
            Text: "Hello, world!"
            Tokens: [15496, 11, 995, 0]
            Breakdown:
              [15496] → "Hello"
              [11]    → ","
              [995]   → " world"
              [0]     → "!"

        Learning Note:
            This helps understand:
            - How BPE splits text
            - Whitespace handling
            - Multi-character tokens
        """
        tokens = self.encode(text)
        stats = self.tokenize_and_count(text)

        # Get individual token strings
        token_strings = []
        for token in tokens:
            token_str = self.decode([token])
            token_strings.append(token_str)

        result = {
            "text": text,
            "tokens": tokens,
            "token_strings": token_strings,
            "stats": stats,
        }

        if verbose:
            print(f'Text: "{text}"')
            print(f"Tokens: {tokens}")
            print(f"Num tokens: {stats['num_tokens']}")
            print(f"Compression: {stats['compression_ratio']:.2f} chars/token")
            print("Breakdown:")
            for token_id, token_str in zip(tokens, token_strings):
                # Escape special characters for display
                display_str = repr(token_str)[1:-1]  # Remove quotes
                print(f"  [{token_id:5d}] → {display_str}")

        return result


# Convenience function for quick testing
def test_tokenizer():
    """
    Quick test of tokenizer functionality.

    Run this to verify tokenizer works correctly.
    """
    print("=" * 60)
    print("Testing Tokenizer (Phase 0.1)")
    print("=" * 60)

    tok = Tokenizer()
    print(f"\n{tok}\n")

    # Test cases
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "The year is 2024.",
        "def hello():\n    print('Hi!')",
    ]

    for text in test_texts:
        print("-" * 60)
        tok.analyze_tokenization(text)
        print()

    print("=" * 60)
    print("✅ Tokenizer test complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_tokenizer()
