# Neural Networks From Scratch — micrograd & makemore deep learning series

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

Other parts of this repository is :
- Attention Mechanism of GPT2 ( Attention Mechanism of LLMs)
- CUDA Programming ( Vectorized Addition and Matrix Multiplication )

---
## Output/s

Example **Micrograd** operation :
```python
x1 = Value(1, label='x1')
w1 = Value(-4.0, label='w1')

x2 = Value(3.75, label='x2')
w2 = Value(2.0, label='w2')

x1w1 = x1 * w1
x1w1.label = 'x1w1'

x2w2 = x2 * w2
x2w2.label = 'x2w2'

loss = (x1w1 - x2w2) ** 2
loss.backward()
draw_dot(loss)
```

Generated Graph with automatic calculated gradients : 

![Micrograd Example](https://github.com/mel-ugaddan/karpathy-makemore-series/blob/main/micrograd_example.png?raw=true)
---

Note : This repository is just my notes, it does not represents how I code professionally. I'm putting it here for now as a blog.

