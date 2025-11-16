"""
BPE (Byte Pair Encoding) Training Implementation

Phase 0.2: Core BPE algorithm in pure Python

Historical Context:
- Algorithm from Gage (1994) for data compression
- Applied to NLP by Sennrich (2015)
- Byte-level variant by GPT-2 (2019)

References:
- ADR-002: BPE Implementation Choices
- nanochat/rustbpe/src/lib.rs
- Papers: Gage (1994), Sennrich (2015)
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
import json


class BPETrainer:
    """
    Byte Pair Encoding trainer.

    Implements the classic BPE algorithm:
    1. Start with byte vocabulary (256 tokens)
    2. Count all adjacent pairs
    3. Merge most frequent pair
    4. Repeat until target vocabulary size

    Example:
        >>> trainer = BPETrainer(vocab_size=512)
        >>> trainer.train(["hello world", "hello"])
        >>> tokens = trainer.encode("hello")
        >>> print(tokens)

    Design (ADR-002):
    - Pure Python (educational, not optimized)
    - Byte-level encoding (handles all UTF-8)
    - Greedy merging (simple, effective)
    - ~10× slower than Rust (acceptable for learning)
    """

    def __init__(self, vocab_size: int = 65536):
        """
        Initialize BPE trainer.

        Args:
            vocab_size: Target vocabulary size (default: 65536 = 2^16, nanochat size)

        Note:
            Vocabulary includes:
            - 256 byte tokens (0-255)
            - (vocab_size - 256) learned merge tokens
        """
        self.vocab_size = vocab_size
        self.merges = []  # List of (pair, new_token_id) tuples
        self.vocab = {}  # token_id → bytes mapping

        # Initialize with byte vocabulary (0-255)
        for i in range(256):
            self.vocab[i] = bytes([i])

    def train(self, texts: List[str], verbose: bool = True) -> Dict:
        """
        Train BPE on a corpus of texts.

        Args:
            texts: List of strings to train on
            verbose: If True, print progress

        Returns:
            Dictionary with training statistics

        Algorithm:
            1. Convert all texts to byte sequences
            2. Iteratively merge most frequent pair
            3. Stop when vocab_size reached

        Complexity:
            O(n * vocab_size) where n = total characters

        Example:
            >>> trainer = BPETrainer(vocab_size=300)
            >>> stats = trainer.train(["hello world"] * 100)
            >>> print(stats['num_merges'])
            44  # 300 - 256 base tokens
        """
        if verbose:
            print(f"Training BPE tokenizer (target vocab: {self.vocab_size})")
            print(f"Training on {len(texts)} texts...")

        # Step 1: Convert texts to byte sequences
        token_sequences = []
        total_bytes = 0
        for text in texts:
            byte_seq = self._encode_bytes(text)
            token_sequences.append(byte_seq)
            total_bytes += len(byte_seq)

        if verbose:
            print(f"Total bytes: {total_bytes:,}")

        # Step 2: Iteratively merge pairs
        num_merges = self.vocab_size - 256  # Number of merges needed
        for merge_idx in range(num_merges):
            # Count all pairs across all sequences
            pair_counts = self._count_pairs(token_sequences)

            if not pair_counts:
                if verbose:
                    print(f"No more pairs to merge. Stopping at {256 + merge_idx} tokens.")
                break

            # Find most frequent pair
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            frequency = pair_counts[most_frequent_pair]

            # Create new token for this pair
            new_token_id = 256 + merge_idx

            # Merge this pair in all sequences
            token_sequences = self._merge_pair_in_all(
                token_sequences, most_frequent_pair, new_token_id
            )

            # Record the merge
            self.merges.append((most_frequent_pair, new_token_id))

            # Update vocab
            pair_bytes = self.vocab[most_frequent_pair[0]] + self.vocab[most_frequent_pair[1]]
            self.vocab[new_token_id] = pair_bytes

            # Progress
            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"  Merge {merge_idx + 1}/{num_merges}: "
                      f"{most_frequent_pair} → {new_token_id} "
                      f"(freq: {frequency:,})")

        if verbose:
            print(f"✅ Training complete! Vocabulary size: {256 + len(self.merges)}")

        return {
            "vocab_size": 256 + len(self.merges),
            "num_merges": len(self.merges),
            "total_bytes": total_bytes,
            "num_texts": len(texts),
        }

    def _encode_bytes(self, text: str) -> List[int]:
        """
        Convert text to list of byte values.

        Args:
            text: Input string

        Returns:
            List of integers (0-255) representing UTF-8 bytes

        Example:
            >>> self._encode_bytes("Hi")
            [72, 105]  # 'H' = 72, 'i' = 105
        """
        return list(text.encode('utf-8'))

    def _count_pairs(self, token_sequences: List[List[int]]) -> Counter:
        """
        Count all adjacent pairs across all sequences.

        Args:
            token_sequences: List of token sequences

        Returns:
            Counter mapping pairs to frequencies

        Example:
            >>> sequences = [[1, 2, 3], [1, 2, 4]]
            >>> self._count_pairs(sequences)
            Counter({(1, 2): 2, (2, 3): 1, (2, 4): 1})
        """
        pairs = Counter()
        for sequence in token_sequences:
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pairs[pair] += 1
        return pairs

    def _merge_pair_in_all(
        self,
        token_sequences: List[List[int]],
        pair: Tuple[int, int],
        new_token_id: int
    ) -> List[List[int]]:
        """
        Merge a pair in all sequences.

        Args:
            token_sequences: List of token sequences
            pair: Pair to merge (left, right)
            new_token_id: ID for merged token

        Returns:
            Updated sequences with pair merged

        Example:
            >>> sequences = [[1, 2, 3], [1, 2, 1, 2]]
            >>> self._merge_pair_in_all(sequences, (1, 2), 256)
            [[256, 3], [256, 256]]
        """
        return [self._merge_pair(seq, pair, new_token_id) for seq in token_sequences]

    def _merge_pair(
        self,
        tokens: List[int],
        pair: Tuple[int, int],
        new_token_id: int
    ) -> List[int]:
        """
        Merge all occurrences of a pair in a sequence.

        Args:
            tokens: Token sequence
            pair: Pair to merge
            new_token_id: Replacement token ID

        Returns:
            New sequence with pairs merged

        Algorithm:
            Scan left to right, replacing (a, b) with new_id

        Example:
            >>> self._merge_pair([1, 2, 3, 1, 2], (1, 2), 256)
            [256, 3, 256]
        """
        if len(tokens) < 2:
            return tokens

        result = []
        i = 0
        while i < len(tokens):
            # Check if we can form the pair
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                result.append(new_token_id)
                i += 2  # Skip both tokens in pair
            else:
                result.append(tokens[i])
                i += 1

        return result

    def encode(self, text: str) -> List[int]:
        """
        Encode text using trained BPE.

        Args:
            text: Input text

        Returns:
            List of token IDs

        Algorithm:
            1. Convert to bytes
            2. Apply merges in order

        Example:
            >>> trainer.train(["hello hello hello"])
            >>> trainer.encode("hello")
            [256]  # Assuming "hello" was merged into single token
        """
        # Start with byte encoding
        tokens = self._encode_bytes(text)

        # Apply each merge in order
        for pair, new_token_id in self.merges:
            tokens = self._merge_pair(tokens, pair, new_token_id)

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded string

        Example:
            >>> tokens = trainer.encode("hello")
            >>> trainer.decode(tokens)
            'hello'
        """
        # Concatenate bytes for each token
        byte_array = bytearray()
        for token_id in tokens:
            if token_id in self.vocab:
                byte_array.extend(self.vocab[token_id])
            else:
                raise ValueError(f"Unknown token ID: {token_id}")

        # Decode UTF-8
        return byte_array.decode('utf-8', errors='replace')

    def save(self, path: str):
        """
        Save trained BPE model to file.

        Args:
            path: File path to save to

        Format:
            JSON file with:
            - vocab_size
            - merges: list of [(pair, token_id), ...]
            - vocab: mapping of token_id → base64(bytes)
        """
        import base64

        # Convert vocab bytes to base64 for JSON
        vocab_serializable = {}
        for token_id, token_bytes in self.vocab.items():
            vocab_serializable[str(token_id)] = base64.b64encode(token_bytes).decode('ascii')

        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "vocab": vocab_serializable,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved BPE model to {path}")

    def load(self, path: str):
        """
        Load trained BPE model from file.

        Args:
            path: File path to load from
        """
        import base64

        with open(path, 'r') as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        # Convert merges back to proper format: ((int, int), int)
        self.merges = [(tuple(pair), token_id) for pair, token_id in data["merges"]]

        # Decode vocab from base64
        self.vocab = {}
        for token_id_str, token_bytes_b64 in data["vocab"].items():
            token_id = int(token_id_str)
            token_bytes = base64.b64decode(token_bytes_b64)
            self.vocab[token_id] = token_bytes

        print(f"✅ Loaded BPE model from {path} ({len(self.merges)} merges)")

    def get_stats(self) -> Dict:
        """
        Get statistics about the trained model.

        Returns:
            Dictionary with model statistics
        """
        return {
            "vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "base_vocab": 256,
            "learned_tokens": len(self.merges),
        }


def train_bpe_from_file(
    input_path: str,
    vocab_size: int = 65536,
    output_path: Optional[str] = None
) -> BPETrainer:
    """
    Convenience function to train BPE from a text file.

    Args:
        input_path: Path to text file
        vocab_size: Target vocabulary size
        output_path: Optional path to save trained model

    Returns:
        Trained BPETrainer

    Example:
        >>> trainer = train_bpe_from_file("corpus.txt", vocab_size=1000)
        >>> trainer.encode("hello world")
    """
    print(f"Loading corpus from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into lines for training (helps with memory)
    texts = [line for line in text.split('\n') if line.strip()]

    trainer = BPETrainer(vocab_size=vocab_size)
    trainer.train(texts)

    if output_path:
        trainer.save(output_path)

    return trainer


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("BPE Training Demo (Phase 0.2)")
    print("=" * 60)

    # Small example
    texts = [
        "hello world",
        "hello",
        "world",
        "hello world hello",
    ] * 10  # Repeat for more data

    trainer = BPETrainer(vocab_size=280)  # 256 + 24 merges
    stats = trainer.train(texts, verbose=True)

    print("\n" + "=" * 60)
    print("Testing encoding/decoding:")
    print("=" * 60)

    test_strings = ["hello", "world", "hello world"]
    for text in test_strings:
        tokens = trainer.encode(text)
        decoded = trainer.decode(tokens)
        print(f"Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Match: {text == decoded}")
        print()

    print("=" * 60)
    print(f"✅ BPE Training Demo Complete!")
    print(f"Stats: {stats}")
    print("=" * 60)
