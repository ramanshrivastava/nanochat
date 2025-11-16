# Learning Guide: Mini-Nanochat

## 🎯 How to Use This Project

This project is designed for **active learning**, not passive reading. Each phase builds incrementally with validation points.

## 📚 Learning Methodology

### 1. **Before Each Commit**

Read the relevant materials:

```bash
# Example for Phase 0.2 (BPE Training)
1. Read ADR-002 in docs/adrs/002-bpe-vs-wordpiece.md
2. Read paper summary in docs/papers/bpe-1994.md
3. Review nanochat reference: /home/user/nanochat/rustbpe/
4. Check comparison: docs/comparisons/tokenizer-comparison.md
```

### 2. **During Implementation**

Understand the code deeply:

```python
# Don't just copy - understand each line
# Ask yourself:
# - Why this approach over alternatives?
# - What would break if I changed this?
# - How does nanochat do it differently?
# - What does the paper say?
```

### 3. **After Each Commit**

Validate your understanding:

```bash
# Run tests
pytest tests/unit/test_tokenizer.py -v

# Complete checkpoint exercises
# See docs/checkpoints/phase0-checkpoint.md

# Try the debugging challenges
# Extend the code with new features
```

## 🗺️ Learning Path

### **Phase 0: Tokenization (Week 1)**

**Commits:** 0.1 → 0.2 → 0.3 → 0.4

**Learning Focus:**
- Subword tokenization fundamentals
- BPE algorithm mechanics
- Data loading basics

**Key Papers:**
- Gage (1994): "A New Algorithm for Data Compression"
- Sennrich (2015): "Neural Machine Translation of Rare Words with Subword Units"
- Radford (2019): GPT-2 - Byte-level BPE

**Checkpoint:** Train tokenizer on 1MB corpus, achieve 4+ chars/token

---

### **Phase 1: Model Architecture (Week 2)**

**Commits:** 1.1 → 1.2 → 1.3 → 1.4 → 1.5

**Learning Focus:**
- Transformer architecture
- Self-attention mechanism
- Position encoding alternatives

**Key Papers:**
- Vaswani (2017): "Attention is All You Need"
- Radford (2018): GPT-1
- Su (2021): RoFormer (RoPE)

**Checkpoint:** Initialize 10M param model, forward pass works

---

### **Phase 2-3: Training & Optimization (Week 3)**

**Commits:** 2.1 → 2.4, 3.1 → 3.3

**Learning Focus:**
- Training loop fundamentals
- Optimizer algorithms
- Mixed precision training

**Key Papers:**
- Kingma (2014): Adam
- Loshchilov (2017): AdamW
- Huh (2024): Muon

**Checkpoint:** Overfit on 1000 sentences, loss < 0.1

---

## 🎓 Study Techniques

### **1. Active Reading**

When reading ADRs:
- ✅ Summarize each section in your own words
- ✅ Draw diagrams of the architecture
- ✅ Note questions and look up answers
- ❌ Don't passively skim

### **2. Code Comparison**

Always compare implementations:

```python
# Mini-nanochat (ours)
def simple_attention(q, k, v):
    scores = q @ k.T / sqrt(d_k)
    weights = softmax(scores)
    return weights @ v

# Full nanochat
# See /home/user/nanochat/nanochat/gpt.py:120-145
# Adds: masking, dropout, multi-head, optimizations

# Industry (PyTorch)
# torch.nn.MultiheadAttention
# Adds: batch processing, fused kernels, flash attention
```

### **3. Paper Mapping**

Connect paper equations to code:

```python
# Paper: "Attention is All You Need" (Vaswani 2017)
# Equation 1: Attention(Q,K,V) = softmax(QK^T/√d_k)V

# Our code (gpt.py:157):
attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
attn_weights = F.softmax(attn_scores, dim=-1)
output = attn_weights @ v
```

### **4. Deliberate Practice**

Complete all exercises:

- **Understanding Exercises (⭐☆☆):** Test comprehension
- **Extension Exercises (⭐⭐☆):** Apply knowledge
- **Debugging Challenges (⭐⭐⭐):** Deep understanding
- **Research Projects (⭐⭐⭐⭐):** Innovation

## 📊 Progress Tracking

### **Self-Assessment Checklist**

After each phase, check if you can:

**Phase 0:**
- [ ] Explain BPE algorithm to someone
- [ ] Implement BPE from scratch (no libs)
- [ ] Debug tokenization issues
- [ ] Compare tokenizers (GPT-2, LLaMA, etc.)

**Phase 1:**
- [ ] Derive attention equation
- [ ] Implement transformer without tutorial
- [ ] Explain RoPE vs learned PE
- [ ] Modify architecture (heads, layers)

**Phase 2-3:**
- [ ] Implement Adam from paper
- [ ] Debug training instabilities
- [ ] Explain AdamW vs Adam
- [ ] Tune hyperparameters

## 🔧 Tools Usage

### **Attention Visualizer**

```bash
# After Phase 1.2
python tools/visualizer/show_attention.py \
    --checkpoint checkpoints/phase1.pt \
    --prompt "The cat sat on the" \
    --layer 2 --head 0
```

### **Training Profiler**

```bash
# After Phase 2.2
python tools/profiler/profile_training.py \
    --config examples/tiny_model.py \
    --steps 100
```

### **Code Comparator**

```bash
# Anytime
python tools/comparator/compare.py \
    --ours mini_nanochat/gpt.py \
    --theirs /home/user/nanochat/nanochat/gpt.py
```

## 📝 Note-Taking Template

For each commit, take notes:

```markdown
# Commit X.Y: [Title]

## What I Learned
- Key concept 1
- Key concept 2

## Design Decisions
- Decision 1: [why?]
- Decision 2: [trade-offs?]

## Comparison
| Aspect | Ours | nanochat | Industry |
|--------|------|----------|----------|
| ... | ... | ... | ... |

## Questions
- Q1: [question]
  - Answer: [after research]

## Exercises Completed
- [x] Exercise 1
- [x] Exercise 2
- [ ] Bonus challenge (in progress)

## Time Spent
- Reading: X hours
- Implementing: X hours
- Debugging: X hours
- Exercises: X hours
```

## 🚨 Common Pitfalls

### **Pitfall 1: Rushing Through**

❌ **Wrong:** Implement all of Phase 0 in one day without understanding
✅ **Right:** Spend 1-2 days per commit, do all exercises

### **Pitfall 2: Skipping Papers**

❌ **Wrong:** "I'll just copy the code"
✅ **Right:** Read paper summary → understand equation → implement

### **Pitfall 3: Not Comparing**

❌ **Wrong:** Only look at our implementation
✅ **Right:** Always compare: ours vs nanochat vs industry

### **Pitfall 4: Passive Learning**

❌ **Wrong:** Just read the code
✅ **Right:** Modify it, break it, fix it, extend it

## 🎯 Success Metrics

### **Week-by-Week Goals**

**Week 1 (Phase 0):**
- Can train BPE tokenizer
- Understand compression ratio
- Complete checkpoint exercises

**Week 2 (Phase 1):**
- Can implement transformer from scratch
- Understand attention mechanism
- Visualize attention patterns

**Week 3 (Phase 2-3):**
- Can overfit on small dataset
- Understand optimizer algorithms
- Debug training issues

**Week 4 (Phase 4):**
- Can train on multiple GPUs
- Understand distributed training
- Measure scaling efficiency

## 💡 Study Tips

### **Time Management**

Recommended schedule per commit:

```
Day 1 (2-3 hours):
- Read ADR + papers
- Review nanochat code
- Study comparisons

Day 2 (2-3 hours):
- Implement the code
- Write tests
- Debug issues

Day 3 (1-2 hours):
- Complete checkpoint exercises
- Do bonus challenges
- Document learnings
```

### **When Stuck**

1. **Re-read the ADR** - Often answers are there
2. **Check nanochat** - See how they solved it
3. **Read the paper** - Go to the source
4. **Debug systematically** - Print shapes, values
5. **Ask questions** - Document what you don't understand

## 🔗 Additional Resources

### **For Phase 0 (Tokenization):**
- [HuggingFace Tokenizers Guide](https://huggingface.co/docs/tokenizers/)
- [SentencePiece Paper](https://arxiv.org/abs/1808.06226)
- [tiktoken Source Code](https://github.com/openai/tiktoken)

### **For Phase 1 (Architecture):**
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention is All You Need (annotated)](http://nlp.seas.harvard.edu/annotated-transformer/)
- [RoFormer Paper](https://arxiv.org/abs/2104.09864)

### **For Phase 2-3 (Training):**
- [Adam Paper](https://arxiv.org/abs/1412.6980)
- [AdamW Paper](https://arxiv.org/abs/1711.05101)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

## ✅ Completion Criteria

You've successfully completed mini-nanochat when you can:

1. **Implement** a 500M param LLM trainer from scratch (no libs except PyTorch)
2. **Explain** every design decision and trade-off
3. **Compare** approaches across nanochat, GPT-2, LLaMA
4. **Debug** training issues independently
5. **Extend** with new features from papers
6. **Teach** concepts to others clearly

---

**Remember:** The goal is not to finish quickly, but to understand deeply.

**Good luck on your learning journey! 🚀**
