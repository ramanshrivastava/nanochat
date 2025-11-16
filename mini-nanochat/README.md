# Mini-Nanochat: Learning-Focused LLM Implementation

**A reasoning-based, historically-grounded implementation of a minimal LLM training system.**

## 🎯 Purpose

This is a learning-focused reimplementation of [nanochat](https://github.com/karpathy/nanochat) built from scratch to deeply understand:

- How LLMs are trained end-to-end
- Design decisions and their trade-offs
- Historical evolution of transformer techniques
- Industry practices vs educational simplifications

## 📚 Learning Approach

Each commit in this repository:

1. **Implements** a specific feature atomically
2. **Explains** the reasoning via Architecture Decision Records (ADRs)
3. **Compares** to full nanochat and industry practices
4. **Connects** to research papers and historical context
5. **Validates** through tests and checkpoints

## 🗂️ Project Structure

```
mini-nanochat/
├── mini_nanochat/          # Core implementation
├── docs/
│   ├── adrs/               # Architecture Decision Records
│   ├── comparisons/        # Mini vs Full vs Industry
│   ├── checkpoints/        # Learning validation points
│   └── papers/             # Paper summaries & mappings
├── tests/                  # Unit, integration, benchmarks
├── examples/               # Training configurations
└── tools/                  # Visualization & debugging
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone <repo>
cd nanochat/mini-nanochat

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run tests
pytest tests/

# See learning guide
cat LEARNING_GUIDE.md
```

## 📅 Implementation Timeline

- **Phase 0 (Week 1):** Tokenization & Data Basics
- **Phase 1 (Week 2):** Model Architecture (GPT Transformer)
- **Phase 2-3 (Week 3):** Training Loop & Optimizers
- **Phase 4 (Week 4):** Distributed Training
- **Phase 5 (Week 4-5):** Data Pipeline
- **Phase 6 (Week 5-6):** Inference Engine (KV Cache)
- **Phase 7 (Week 6):** Evaluation & Benchmarks
- **Phase 8 (Week 7):** Supervised Fine-Tuning
- **Phase 9 (Week 7-8):** Production Serving

## 📖 Documentation

- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - How to use this project for learning
- **[HISTORICAL_TIMELINE.md](HISTORICAL_TIMELINE.md)** - Maps commits to LLM history
- **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** - Detailed phase breakdown
- **[docs/adrs/](docs/adrs/)** - Architecture decisions with rationale

## 🎓 Learning Outcomes

After completing this project, you will:

- ✅ Understand every component of LLM training
- ✅ Implement transformers from scratch
- ✅ Debug training issues confidently
- ✅ Read research papers and map to code
- ✅ Compare approaches (GPT-2, LLaMA, etc.)
- ✅ Make informed architecture decisions

## 🔗 References

- **Original nanochat:** `/home/user/nanochat`
- **Key Papers:** See [docs/papers/reading-list.md](docs/papers/reading-list.md)
- **Industry Implementations:** GPT-2, LLaMA, etc.

## 📊 Comparison to Full Nanochat

| Aspect | Mini-nanochat | Full nanochat |
|--------|---------------|---------------|
| **Purpose** | Learning-focused | Production-capable |
| **Lines of Code** | ~5,000-7,000 (target) | ~7,859 |
| **Documentation** | Extensive (ADRs, comparisons) | Good (README, comments) |
| **Complexity** | Simplified for clarity | Optimized for performance |
| **Error Handling** | Basic | Comprehensive |
| **Performance** | ~70-80% | 100% (baseline) |

## 📝 License

MIT License (same as nanochat)

## 🙏 Acknowledgments

- Andrej Karpathy for the original [nanochat](https://github.com/karpathy/nanochat)
- The PyTorch team
- The research community behind transformers

---

**Status:** 🚧 In Progress - Phase 0 (Tokenization)

**Current Commit:** Phase 0.1 - Project Setup + Basic Tokenizer
