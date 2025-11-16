# ADR-002: BPE Training Implementation

**Status:** Accepted
**Date:** 2025-01-16
**Commit:** Phase 0.2
**nanochat Reference:** `rustbpe/src/lib.rs:1-500`
**Related Papers:**
- Gage (1994): "A New Algorithm for Data Compression"
- Sennrich (2015): "Neural Machine Translation of Rare Words with Subword Units"

**Industry Comparison:** GPT-2 (tiktoken), nanochat (rustbpe), minbpe (Karpathy)

---

## Context

### Problem Statement

We need to **train** a custom BPE tokenizer that:

1. **Builds vocabulary** from scratch on text corpus
2. **Learns merge rules** to create subword tokens
3. **Produces 65K vocab** (matching nanochat)
4. **Works offline** (no network dependencies)
5. **Is understandable** (educational implementation)

### Current State

Phase 0.1 gave us a tokenizer wrapper, but:
- ❌ Depends on network (HuggingFace download)
- ❌ Can't train custom vocabularies yet
- ❌ Uses pre-built GPT-2 vocab (50K)
- ❌ No understanding of BPE internals

### Constraints

- **Educational:** Must be readable Python (not optimized Rust)
- **Self-contained:** No external vocab files needed
- **Comparable:** Should produce similar results to nanochat's rustbpe
- **Testable:** Unit tests for each step of algorithm

---

## Decision

**Implement BPE training in pure Python with these components:**

1. **Byte-level encoding** (like GPT-2, nanochat)
2. **Iterative merging** (classic BPE algorithm)
3. **Priority queue** for efficient merge selection
4. **Save/load** vocabulary to disk
5. **Compatible** with tiktoken for inference

### Implementation Structure

```python
# mini_nanochat/bpe.py

class BPETrainer:
    def __init__(self, vocab_size=65536):
        self.vocab_size = vocab_size
        self.merges = []  # List of (pair, new_token_id)

    def train(self, texts: List[str]) -> Dict:
        # 1. Convert text to bytes
        # 2. Count pair frequencies
        # 3. Merge most frequent pair
        # 4. Repeat until vocab_size reached

    def save(self, path: str):
        # Save merges and vocab

    def load(self, path: str):
        # Load trained vocabulary
```

---

## Rationale

### Algorithm Overview

**BPE (Byte Pair Encoding) in 4 steps:**

```
Step 1: Start with byte vocabulary (256 tokens)
  Text: "low" → bytes: [108, 111, 119]

Step 2: Count all adjacent pairs
  Pairs: (108,111), (111,119)
  Frequencies: {(108,111): 5, (111,119): 3, ...}

Step 3: Merge most frequent pair
  Most frequent: (108,111) appears 5 times
  Create new token: 256 → represents [108,111]
  Update: "low" → [256, 119]

Step 4: Repeat until vocab_size reached
  After many merges:
  "low" → [single_token_for_low]
```

### Why This Approach?

**1. Byte-level (not character-level)**

```python
# Character-level (BAD - can't handle all Unicode)
"hello 世界" → ['h','e','l','l','o',' ','世','界']  # 世界 = UNK?

# Byte-level (GOOD - handles everything)
"hello 世界" → [104,101,108,108,111,32,228,184,150,231,149,140]
```

**Benefit:** Can represent ANY text, no UNK token needed

**2. Iterative merging (greedy algorithm)**

At each step:
- Count all pairs in current data
- Merge most frequent pair
- Update all occurrences
- Repeat

**Complexity:** O(n * vocab_size) where n = corpus size

**3. Priority queue for efficiency**

```python
# Naive approach: O(n) scan each iteration
for iteration in range(vocab_size - 256):
    pairs = count_all_pairs(data)  # O(n)
    most_frequent = max(pairs)      # O(vocab)

# Better: Use heap/priority queue
heap = MaxHeap(pair_counts)
while len(vocab) < vocab_size:
    most_frequent = heap.pop()      # O(log vocab)
```

**Speedup:** ~10× faster for large corpora

---

## nanochat Comparison

### What nanochat Does (rustbpe)

```rust
// rustbpe/src/lib.rs:100-300
pub fn train(text: &str, vocab_size: usize) -> Vocab {
    // 1. Regex split (GPT-4 pattern)
    // 2. Byte-level encoding
    // 3. BPE merging with parallel counting
    // 4. Return mergeable_ranks
}
```

**Key features:**
- **Rust implementation** (~1000 lines)
- **Regex pre-tokenization** (GPT-4 pattern)
- **Parallel pair counting** (rayon)
- **Training speed:** ~1 minute for 1B chars

### What We're Implementing

| Aspect | Mini-nanochat | nanochat (rustbpe) | Difference |
|--------|---------------|-------------------|------------|
| **Language** | Python | Rust | Educational vs Performance |
| **Lines** | ~200 | ~1000 | Simplified |
| **Regex split** | Phase 0.3 | Yes | Add later |
| **Parallelism** | No | Yes (rayon) | Single-threaded for now |
| **Speed** | ~10 min for 1B chars | ~1 min | 10× slower (acceptable for learning) |
| **Output** | Compatible | tiktoken format | Same format |

---

## Algorithm Implementation

### Detailed Steps

**Step 1: Byte-level encoding**

```python
def encode_bytes(text: str) -> List[int]:
    """Convert text to list of byte values."""
    return list(text.encode('utf-8'))

# Example:
encode_bytes("Hello") → [72, 101, 108, 108, 111]
```

**Step 2: Count pair frequencies**

```python
def count_pairs(tokens: List[int]) -> Dict[Tuple[int,int], int]:
    """Count all adjacent pairs."""
    pairs = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1])
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

# Example:
tokens = [72, 101, 108, 108, 111]  # "Hello"
# Pairs: (72,101), (101,108), (108,108), (108,111)
```

**Step 3: Merge most frequent**

```python
def merge_pair(tokens: List[int], pair: Tuple[int,int], new_id: int) -> List[int]:
    """Replace all occurrences of pair with new_id."""
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            result.append(new_id)
            i += 2  # Skip both tokens
        else:
            result.append(tokens[i])
            i += 1
    return result

# Example:
tokens = [108, 108, 111]  # "llo"
merge_pair(tokens, (108,108), 256)  → [256, 111]
```

**Step 4: Iterate until vocab_size**

```python
def train_bpe(text: str, vocab_size: int) -> List[Tuple]:
    tokens = encode_bytes(text)
    merges = []

    for i in range(256, vocab_size):  # 256 base + merges
        pairs = count_pairs(tokens)
        if not pairs:
            break

        # Most frequent pair
        most_freq = max(pairs.items(), key=lambda x: x[1])
        pair, count = most_freq

        # Merge it
        tokens = merge_pair(tokens, pair, i)
        merges.append((pair, i))

    return merges
```

---

## Trade-offs

### Benefits ✅

1. **Understandable**
   - Pure Python, ~200 lines
   - Each step is clear
   - Can step through with debugger

2. **Self-contained**
   - No network dependencies
   - No external files needed
   - Works offline

3. **Educational**
   - See exactly how BPE works
   - Understand merge decisions
   - Visualize vocabulary growth

4. **Compatible**
   - Output format matches tiktoken
   - Can use with nanochat
   - Interchangeable with rustbpe

### Limitations ❌

1. **Speed**
   - Python: ~10 min for 1B chars
   - Rust: ~1 min for 1B chars
   - **10× slower** (acceptable for learning)

2. **Memory**
   - Stores all tokens in RAM
   - No streaming (for now)
   - Limit: ~1-2B characters

3. **No parallelism**
   - Single-threaded
   - Can't use multiple cores
   - (Could add with multiprocessing later)

4. **No regex pre-tokenization yet**
   - Phase 0.2: Basic BPE only
   - Phase 0.3: Add GPT-4 regex pattern
   - Incrementally add complexity

### Performance Impact

| Corpus Size | Python BPE | Rust BPE | Ratio |
|-------------|-----------|----------|-------|
| **1M chars** | ~1 second | ~0.1 sec | 10× |
| **100M chars** | ~1 minute | ~6 sec | 10× |
| **1B chars** | ~10 min | ~1 min | 10× |

**Acceptable?** Yes for learning! We're optimizing for understanding, not speed.

---

## Historical Evolution

### BPE Timeline

```
1994: Gage - Data compression
  - Original BPE for file compression
  - Greedy algorithm, byte pairs

2015: Sennrich - Neural MT
  - Applied to NLP
  - Subword units for translation
  - Character-level BPE

2016: Google - WordPiece (BERT)
  - Similar to BPE
  - Likelihood-based merging

2019: OpenAI - Byte-level BPE (GPT-2)
  - Bytes not characters
  - No UNK token
  - 50K vocabulary

2024: nanochat - GPT-4 style
  - 65K vocabulary
  - Regex pre-tokenization
  - Rust implementation

2025: mini-nanochat - Educational
  - Python implementation
  - Step-by-step learning
```

### Why BPE Stuck Around

Despite being from 1994, BPE is still used because:

1. **Simple** - Easy to understand and implement
2. **Effective** - Good compression ratio
3. **Universal** - Byte-level handles all text
4. **Fast inference** - O(n) encoding/decoding
5. **No OOV** - Can represent any string

---

## Learning Outcomes

After implementing Phase 0.2, you will understand:

1. **BPE Algorithm Mechanics**
   - How pairs are counted
   - Why greedy merging works
   - Vocabulary growth process

2. **Byte-level Encoding**
   - Why bytes vs characters
   - UTF-8 handling
   - No UNK token guarantee

3. **Trade-offs**
   - Speed: Python vs Rust
   - Memory: In-RAM vs streaming
   - Vocab size: 50K vs 65K vs 100K

4. **Implementation Details**
   - Data structures (heap, dict)
   - Algorithm complexity
   - Optimization opportunities

5. **Comparison Skills**
   - Our Python vs nanochat's Rust
   - BPE vs WordPiece
   - Different vocab sizes

---

## References

### Papers

- **Gage (1994):** "A New Algorithm for Data Compression"
  - Original BPE for file compression
  - Core algorithm we implement

- **Sennrich (2015):** "Neural Machine Translation of Rare Words with Subword Units"
  - BPE applied to NLP
  - Explains why subword units work
  - [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)

### Code

- **nanochat:** `/home/user/nanochat/rustbpe/src/lib.rs`
  - Production Rust implementation
  - ~1000 lines, optimized

- **minbpe (Karpathy):** [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
  - Minimal BPE in Python (~100 lines)
  - Great reference implementation

- **tiktoken:** [https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)
  - Fast C++ inference
  - We'll use for encoding after training

### Blog Posts

- "Let's build GPT: from scratch" - Andrej Karpathy
  - YouTube video covering tokenization
  - Explains BPE visually

---

## Exercises

### Exercise 1: Manual BPE ⭐☆☆

**Task:** Manually perform BPE on small text

```
Text: "low low low low lower lower newest newest newest newest newest newest"

Step 0: Convert to bytes (simplified - use letters as "bytes")
  [l,o,w, ,l,o,w, ,l,o,w, ,l,o,w, ,l,o,w,e,r,...]

Step 1: Count pairs
  Most frequent: ???

Step 2: Merge most frequent
  New vocab token: 256 → ???

Continue for 3 more merges...

Questions:
1. What are the first 3 merges?
2. What's the final encoding of "lowest"?
3. Why did BPE choose these merges?
```

**Time:** 20 minutes
**Learning:** Understand merge selection

---

### Exercise 2: Implement Simple BPE ⭐⭐☆

**Task:** Implement basic BPE (no optimizations)

```python
def simple_bpe(text: str, num_merges: int) -> List[Tuple]:
    """
    Implement BPE with just the core algorithm.
    Return list of merges performed.
    """
    # Your implementation here
    pass

# Test it
merges = simple_bpe("low low low lower", num_merges=5)
print(merges)  # Should show which pairs were merged
```

**Time:** 45 minutes
**Hint:** Start with byte encoding, then count pairs, then merge

---

### Exercise 3: Analyze Vocabulary ⭐⭐⭐

**Task:** Train BPE on different texts, compare vocabularies

```python
# Train on English prose
merges_english = train_bpe(english_text, vocab_size=1000)

# Train on Python code
merges_code = train_bpe(python_code, vocab_size=1000)

# Train on JSON data
merges_json = train_bpe(json_data, vocab_size=1000)

# Compare:
# - Which has more multi-byte tokens?
# - Which compresses better?
# - How do vocabularies differ?
```

**Time:** 60 minutes
**Learning:** See how domain affects vocabulary

---

## Discussion Questions

1. **Why greedy merging?**
   - Why not look ahead?
   - Is greedy optimal?
   - What could go wrong?

2. **Vocabulary size trade-offs**
   - GPT-2: 50K, nanochat: 65K, GPT-4: 100K
   - Larger vocab = shorter sequences (good)
   - Larger vocab = more embeddings (bad)
   - What's optimal?

3. **Byte-level vs character-level**
   - Why did GPT-2 switch to bytes?
   - What about Chinese/Japanese?
   - Performance implications?

4. **Alternative algorithms**
   - WordPiece (BERT)
   - Unigram LM (SentencePiece)
   - Why is BPE still popular?

5. **Python vs Rust implementation**
   - When is 10× slower acceptable?
   - When do you need Rust?
   - Cost-benefit of optimization?

---

## Next Steps

**Phase 0.3:** Add regex pre-tokenization (GPT-4 pattern)
- Understand why pre-splitting matters
- Implement GPT-4 regex pattern
- Compare with/without pre-tokenization

**Phase 0.4:** Validate and compare
- Train on real corpus (FineWeb sample)
- Compare output to nanochat's rustbpe
- Measure compression ratios

---

**This ADR captures the reasoning behind our BPE implementation. Understanding this will make the code obvious! 🎯**
