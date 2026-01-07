# Neural Networks From Scratch — micrograd & makemore

This repository contains implementations, notes, and experiments based on **Andrej Karpathy’s “Neural Networks: Zero to Hero” series**, focusing on:

- **micrograd** — a tiny automatic differentiation engine
- **makemore** — character-level language models built step by step

The goal is to deeply understand how neural networks work internally, from backpropagation to language modeling.

---

## Projects

### micrograd
A minimal autograd engine that builds and differentiates scalar-valued computation graphs.

**What it covers**
- Reverse-mode automatic differentiation
- Backpropagation via the chain rule
- Manual construction of neural networks
- Gradient-based optimization

**Key concepts**
- Computational graphs (DAGs)
- Gradient accumulation
- Parameter updates
- MLPs from first principles
### makemore
A character-level language modeling project that incrementally builds modern NLP concepts from scratch.

**Models included**
1. Bigram count-based models
2. Neural bigram models
3. Multi-layer perceptrons (MLPs)
4. Batch normalization
5. PyTorch training loops
6. Sampling and evaluation

**Key concepts**
- Character tokenization
- Embeddings
- Softmax and cross-entropy loss
- Logits and probability distributions
- Training stability and initialization

This foundation is especially useful for work involving:
- Language models (LLMs)
- BERT-style architectures
- Noisy-label learning
- Representation and embedding analysis
