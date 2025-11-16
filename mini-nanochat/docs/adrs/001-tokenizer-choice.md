# ADR-001: Tokenizer Choice and Design

**Status:** Accepted
**Date:** 2025-01-16
**Commit:** Phase 0.1
**nanochat Reference:** `nanochat/tokenizer.py:1-80`
**Related Papers:**
- Gage (1994): "A New Algorithm for Data Compression"
- Sennrich (2015): "Neural Machine Translation of Rare Words with Subword Units"
- GPT-2 (2019): Byte-level BPE

**Industry Comparison:** GPT-2 (tiktoken), LLaMA (SentencePiece), GPT-4 (tiktoken)

---

## Context

### Problem Statement

We need a tokenizer that converts raw text into integer tokens that our model can process. The tokenizer must:

1. **Handle arbitrary text** (including rare words, code, multiple languages)
2. **Be efficient** (fast encoding/decoding)
3. **Be learnable** (good for our educational goals)
4. **Match nanochat** (for comparison purposes)

### Constraints

- **Computational:** Training should take minutes, not hours
- **Educational:** Should be understandable, not a black box
- **Compatibility:** Should work with existing tools (tiktoken)
- **Quality:** Compression ratio ~4-5 characters per token

---

## Decision

**We will use a two-component approach:**

1. **Training:** Start with existing tokenizer (tiktoken) for Phase 0.1
2. **Later (Phase 0.2):** Implement custom BPE trainer in Python
3. **Inference:** Always use tiktoken (fast C++ implementation)

**Initial Implementation (Phase 0.1):**

```python
# mini_nanochat/tokenizer.py
import tiktoken

class Tokenizer:
    def __init__(self, vocab_size=65536):
        # Use GPT-2 tokenizer as baseline
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
        return self.enc.decode(tokens)

    def __len__(self):
        return self.vocab_size
```

---

## Rationale

### Why This Approach?

**1. Start Simple, Add Complexity Later**

- Phase 0.1: Use existing tokenizer (tiktoken) ✅
  - Get started immediately
  - Focus on understanding tokenization concepts
  - Validate data pipeline first

- Phase 0.2: Implement BPE training ✅
  - Understand algorithm deeply
  - Control vocabulary exactly
  - Match nanochat approach

**2. Separate Training from Inference**

nanochat does this:
- **Training:** rustbpe (Rust implementation, fast)
- **Inference:** tiktoken (C++ implementation, 10-100× faster)

We'll do similarly:
- **Training:** Python BPE (easier to understand)
- **Inference:** tiktoken (reuse fast implementation)

**3. Educational Value**

Using tiktoken initially lets us:
- ✅ Understand tokenization output
- ✅ Test the full pipeline quickly
- ✅ Compare different tokenizers easily
- ✅ Focus on model/training first

Implementing BPE later lets us:
- ✅ Understand the algorithm deeply
- ✅ Control vocabulary size/content
- ✅ See trade-offs (speed vs customization)

### Alternatives Considered

| Alternative | Pros | Cons | Why Rejected |
|-------------|------|------|--------------|
| **HuggingFace Tokenizers** | Feature-rich, well-tested | Complex API, many features we don't need | Too much abstraction for learning |
| **SentencePiece** | Language-agnostic, unigram LM option | C++ dependency, complex configuration | Harder to understand internals |
| **Character-level** | No OOV, simple | Sequences 4-5× longer, harder to learn | Poor efficiency |
| **Word-level** | Simple, interpretable | Huge vocab, OOV problems | Doesn't scale |
| **Custom from scratch (Phase 0.1)** | Full control, deep learning | Slow, reinventing wheel | Not time-efficient for start |

---

## nanochat Comparison

### What nanochat Does

```python
# nanochat/tokenizer.py:8-50
class Tokenizer:
    def __init__(self, vocab_path=None):
        if vocab_path is None:
            # Use default GPT-4 style tokenizer
            vocab_path = os.path.join(base_dir(), "tok65536.model")

        # Load tiktoken encoding
        mergeable_ranks = load_tiktoken_bpe(vocab_path)
        self.enc = tiktoken.Encoding(
            name="nanochat",
            pat_str=GPT4_SPLIT_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens={...}
        )
```

**File:** `nanochat/tokenizer.py:8-50`

**Key Differences:**

1. nanochat loads custom vocabulary (65K tokens)
2. Uses GPT-4 splitting pattern
3. Has special tokens (<|endoftext|>, <|python_start|>, etc.)

### What We're Doing Differently (Phase 0.1)

| Aspect | Mini-nanochat | Full nanochat | Why Different |
|--------|---------------|---------------|---------------|
| **Vocab** | GPT-2 (50K) | Custom (65K) | Simpler start, will train custom later |
| **Special tokens** | None yet | 8 special tokens | Add in Phase 8 (SFT) |
| **Split pattern** | GPT-2 default | GPT-4 style | Match later when we train BPE |
| **Implementation** | Wrapper class | Wrapper + custom vocab | Incremental complexity |

---

## Historical Evolution

### Timeline

```
1994: BPE for data compression (Gage)
      ↓
2015: BPE for neural machine translation (Sennrich)
      - Handles rare words
      - Subword units
      ↓
2016: Google's WordPiece (used in BERT)
      - Similar to BPE
      - Likelihood-based merging
      ↓
2018: SentencePiece (Google)
      - Language-agnostic
      - Unigram LM option
      ↓
2019: GPT-2 byte-level BPE (OpenAI)
      - No UNK token
      - Universal (all UTF-8)
      - 50K vocabulary
      ↓
2023: GPT-4 tokenizer (OpenAI)
      - Improved splitting
      - Better number handling
      - ← nanochat uses this style
      ↓
2024: nanochat (Karpathy)
      - GPT-4 style
      - 65K vocabulary
      - rustbpe for training
      ↓
2025: mini-nanochat (This Project)
      - Start with GPT-2 (tiktoken)
      - Implement BPE in Phase 0.2
```

### Why Byte-Level BPE?

**Traditional BPE:**
```python
# Problem: Unknown words
vocab = ["hello", "world", "hel", "lo", "wor", "ld"]
encode("hello world") → [0, 1]  # OK
encode("goodbye")      → [UNK]  # BAD!
```

**Byte-Level BPE (GPT-2):**
```python
# Solution: Every byte is in vocab (0-255)
# Can represent ANY string
vocab = [0, 1, ..., 255, "hello", "world", ...]
encode("hello world") → [token_hello, token_world]
encode("goodbye")     → [token_go, token_od, token_bye]  # NO UNK!
```

**Key Insight:** Byte-level BPE has a guaranteed base vocabulary (256 bytes), so it can represent ANY text.

---

## Trade-offs

### Benefits ✅

1. **No OOV (Out-of-Vocabulary) problem**
   - Can encode any text
   - No UNK tokens

2. **Compression**
   - ~4-5 characters per token
   - Much better than character-level (1 char/token)

3. **Flexible vocabulary size**
   - Can trade off vocab size vs sequence length
   - nanochat uses 65K (more than GPT-2's 50K)

4. **Language-agnostic**
   - Works for all languages
   - Works for code, JSON, etc.

### Limitations ❌

1. **Tokenization is not reversible**
   - "hello world" vs "hello  world" (2 spaces) might tokenize differently
   - Whitespace handling is tricky

2. **Number handling**
   ```python
   # Numbers split differently based on context
   "1234"  → ["1234"]      # Single token
   "12345" → ["12", "345"] # Two tokens (if 12345 not in vocab)
   ```

3. **Vocabulary size trade-off**
   - Small vocab (10K): Longer sequences, less memory
   - Large vocab (100K): Shorter sequences, more memory
   - Sweet spot: 50K-100K

4. **Training time**
   - BPE training on 1B chars: ~10 minutes (Python), ~1 minute (Rust)

### Performance Impact

| Metric | Value | Comparison |
|--------|-------|------------|
| **Encoding speed** | ~1M tokens/sec | (tiktoken, C++) |
| **Training time** | ~10 min for 1B chars | (Python BPE, Phase 0.2) |
| **Vocab size** | 50K (GPT-2) → 65K (nanochat) | 30% more tokens |
| **Compression** | ~4.5 chars/token | (English text) |

---

## Learning Outcomes

After implementing Phase 0.1, you should be able to:

1. **Explain** why subword tokenization beats word-level and character-level
2. **Use** tiktoken to encode/decode text
3. **Understand** byte-level BPE prevents OOV issues
4. **Analyze** tokenization output (what splits where)
5. **Compare** different tokenizers (GPT-2, GPT-4, SentencePiece)

After Phase 0.2 (BPE implementation), you should be able to:

1. **Implement** BPE algorithm from scratch
2. **Train** a custom vocabulary
3. **Tune** vocabulary size based on requirements
4. **Debug** tokenization issues
5. **Optimize** BPE training speed

---

## References

### Papers

- **Gage (1994):** "A New Algorithm for Data Compression"
  - Original BPE for compression
  - Key insight: Repeatedly merge most frequent byte pairs

- **Sennrich (2015):** "Neural Machine Translation of Rare Words with Subword Units"
  - Applied BPE to NMT
  - Handles rare words gracefully
  - [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)

- **GPT-2 (2019):** "Language Models are Unsupervised Multitask Learners"
  - Section 2.2: Byte-level BPE
  - No UNK token, universal encoding
  - [https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

### Code

- **nanochat:** `/home/user/nanochat/nanochat/tokenizer.py`
- **tiktoken:** [https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)
- **SentencePiece:** [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

### Blog Posts / Tutorials

- "Let's build GPT: from scratch, in code, spelled out" - Andrej Karpathy
  - YouTube video covering tokenization
- "minbpe" - Minimal BPE implementation by Karpathy
  - [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)

---

## Exercises

### Exercise 1: Understanding Tokenization ⭐☆☆

**Task:** Tokenize the following texts and analyze the output

```python
from mini_nanochat.tokenizer import Tokenizer

tok = Tokenizer()

texts = [
    "Hello, world!",
    "Hello,  world!",  # Two spaces
    "The year is 2024",
    "The year is 20241",  # Extra digit
    "def hello(): print('hi')",  # Code
]

for text in texts:
    tokens = tok.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Length: {len(tokens)} tokens")
    print(f"Decoded: {tok.decode(tokens)}")
    print()
```

**Questions:**
1. How does whitespace affect tokenization?
2. How are numbers handled?
3. How does code tokenize compared to natural language?

**Time:** 15 minutes

---

### Exercise 2: Compression Analysis ⭐⭐☆

**Task:** Measure compression ratio for different types of text

```python
import os

def compression_ratio(tokenizer, text):
    tokens = tokenizer.encode(text)
    return len(text) / len(tokens)

# Test on different text types:
# 1. English prose
# 2. Python code
# 3. JSON data
# 4. Random characters

# Calculate chars/token for each
# Which compresses best? Why?
```

**Expected Results:**
- English: ~4-5 chars/token
- Code: ~3-4 chars/token (more symbols)
- JSON: ~2-3 chars/token (lots of syntax)

**Time:** 30 minutes

---

### Exercise 3: Tokenizer Comparison ⭐⭐⭐

**Task:** Compare GPT-2 tokenizer (ours) with others

```python
import tiktoken

# GPT-2 (ours)
gpt2 = tiktoken.get_encoding("gpt2")

# GPT-4 (improved)
gpt4 = tiktoken.get_encoding("cl100k_base")

test_text = "The number 12345 appears in the year 2024."

# Compare:
# 1. Token IDs
# 2. Number of tokens
# 3. How numbers split

# Questions:
# - Which handles numbers better?
# - Which is more efficient?
# - Why does GPT-4 use a different pattern?
```

**Time:** 45 minutes
**Hint:** Check how GPT-4's pattern handles numbers differently

---

## Discussion Questions

1. **Why 50K-100K vocabulary size?**
   - Why not 10K (smaller)?
   - Why not 1M (larger)?
   - What's the trade-off?

2. **Character-level vs BPE vs word-level**
   - When would you use each?
   - What are the failure modes?

3. **Multilingual considerations**
   - Does BPE work equally well for all languages?
   - What about Chinese/Japanese (no spaces)?
   - How does vocabulary allocation work?

4. **Code vs natural language**
   - Should you train separate tokenizers?
   - How does code affect vocabulary?

5. **Future directions**
   - What comes after BPE?
   - Why hasn't tokenization changed much since 2019?
   - What problems remain unsolved?

---

## Next Steps

**Phase 0.2:** Implement BPE training algorithm
- Understand merge operations
- Implement training loop
- Train custom 65K vocabulary
- Compare to nanochat's rustbpe

**Phase 0.3:** Build tiny text dataset loader
- Download FineWeb shard
- Tokenize on-the-fly
- Measure loading speed

---

**This ADR captures the reasoning behind our tokenizer choice. Understanding this deeply will pay dividends throughout the project! 🎯**
