# Neural Networks From Scratch — micrograd & makemore ( Intro to Deep Learning series )

This repository contains implementations, notes, and experiments based on **Andrej Karpathy’s “Neural Networks: Zero to Hero” series**, focusing on:

- **micrograd** — a tiny automatic differentiation engine
- **makemore** — character-level language models built step by step

The goal is to deeply understand how pytorch works internally, from backpropagation to language modeling.

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

**Topics included**
1. Bigram count-based models
2. Neural bigram models
3. Multi-layer perceptrons (MLPs)
4. Batch normalization
5. PyTorch training loops
6. Sampling and evaluation
7. Attention Mechanism ( GPT vs BERT )
8. Tokenization (BPE Algorithm)

**Key concepts**
- Character tokenization
- Embeddings
- Softmax and cross-entropy loss
- Logits and probability distributions
- Training stability and initialization

This foundation is especially useful for work involving:
- Language models (LLMs)
- BERT-style architectures
- Representation and embedding analysis

Other parts of this repository is :
- CUDA Programming ( Vectorized Addition and Matrix Multiplication )

---

## Outputs : 

### Example output from **Micrograd** :

#### Operation :
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

#### Generated graph structure with calculated gradients : 

![Micrograd Example](https://github.com/mel-ugaddan/karpathy-makemore-series/blob/main/micrograd_example.png?raw=true)

### Batchnorm Computation :

#### Formula :
$$
\begin{aligned}
\mu &= \frac{1}{m} \sum_{i=1}^{m} x_i \\
\sigma^2 &= \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2 \\
\hat{x}_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} \\
y_i &= \gamma \hat{x}_i + \beta
\end{aligned}
$$


#### Batchnorm computational graph : 

<p align="center">
  <img src="https://github.com/mel-ugaddan/karpathy-makemore-series/blob/derivation/computational_graph_batchnorm.jpg?raw=true" alt="Micrograd Example">
</p>


#### Backpropagate, then compute :

$$
\frac{dL}{dx_i},\frac{dL}{d\sigma^2},\frac{dL}{d\mu}
$$

#### For node <span style="background-color:#c8facc;">(1)</span>, for each element $i$ from vector $\hat{x}$ we have the following  :

$$
\begin{aligned}
\frac{dL}{d\hat{x}_i}&=\frac{dL}{dy_i}\cdot\gamma
\end{aligned}
$$

#### Moving on node <span style="background-color:#c8facc;">(2)</span>, we have $\frac{dL}{d\sigma^2}$ we have the following  :

$$
\begin{aligned}
\frac{dL}{d\sigma^2}&=\sum_{i=1}^{m}\frac{dL}{d\hat{x}_i}\cdot\frac{d\hat{x}_i}{d\sigma^2}\\
&=\gamma\sum_{i=1}^{m}\frac{d}{d\sigma^2}\left[\frac{x_i-\mu}{(\sigma^2+\epsilon)^{\frac{1}{2}}}\right]\cdot\frac{d}{dy_i}\\
&=-\frac{\gamma}{2} \sum_{i=1}^{m}\frac{d}{dy_i}(x_i-\mu)(\sigma^2+\epsilon)^{-\frac{3}{2}}\\
\end{aligned}
$$

$$
\begin{aligned}
\frac{d\sigma^2}{d\mu}&=\frac{d}{d\mu}\left[\frac{1}{m-1}\sum_{i=1}^{m}(x_i-\mu)^2\right]\\
&=-\frac{2}{m-1}\left[\sum_{i=1}^{m}(x_i-\mu)\right]\\
&=-\frac{2}{m-1}\left[(\sum_{i=1}^{m}x_i-\sum_{i=1}^{m}\mu)\right]\\
&=-\frac{2}{m-1}\left[(mx_i-m\mu)\right]\\
&=0
\end{aligned}
$$

$$
\begin{aligned}
\frac{dL}{d\sigma^2}&=\left(\sum_{i=1}^{m}\frac{dL}{d\hat{x}_i}\cdot\frac{d\hat{x}_i}{d\mu}\right)+\left(\frac{dL}{d\sigma^2}\cdot\frac{d\sigma^2}{d\mu}\right)\\
\end{aligned}
$$



---

Note : This repository is just my notes, it does not represents how I code professionally. I'm putting it here for now as a blog.

