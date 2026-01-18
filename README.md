# Neural Networks From Scratch — micrograd & makemore ( Intro to Deep Learning Series )

This repository contains implementations, notes, and experiments based on **Andrej Karpathy’s “Neural Networks: Zero to Hero” series**, focusing on:

- **micrograd** — A tiny automatic differentiation engine : Basically Pytorch from scratch.
- **makemore** — A youtube series about concepts on Deep Learning. 

The goal is to deeply understand how pytorch works internally, from backpropagation to language modeling like GPT or BERT.

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

### Example output from **Makemore Series**, manual backprogpagation of Batchnorm :

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

#### Code :
```
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
```

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

Code :

```
dhpreact = (1-h**2)*dh #  (1-h**2) is from TanH
```

#### Moving on node (2), we have $\frac{dL}{d\sigma^2}$ we have the following  :

$$
\begin{aligned}
\frac{dL}{d\sigma^2}&=\sum_{i=1}^{m}\frac{dL}{d\hat{x}_i}\cdot\frac{d\hat{x}_i}{d\sigma^2}\\
&=\gamma\sum_{i=1}^{m}\frac{d}{d\sigma^2}\left[\frac{x_i-\mu}{(\sigma^2+\epsilon)^{\frac{1}{2}}}\right]\cdot\frac{d}{dy_i}\\
&=-\frac{\gamma}{2} \sum_{i=1}^{m}\frac{d}{dy_i}(x_i-\mu)(\sigma^2+\epsilon)^{-\frac{3}{2}}\\
\end{aligned}
$$

Code :

```
dbnvar_inv = (bndiff * dbnraw).sum(0,keepdims=True)
dbndiff = (torch.ones_like(bndiff)*bnvar_inv)* dbnraw
dbnvar = (-0.5*(bnvar+ 1e-5)**(-3/2)) * dbnvar_inv
```

#### Moving on node (3) :

$$
\begin{aligned}
\frac{dL}{d\mu}&=\left(\sum_{i=1}^{m}\frac{dL}{d\hat{x}_i}\cdot\frac{d\hat{x}_i}{d\mu}\right)+\left(\frac{dL}{d\sigma^2}\cdot\frac{d\sigma^2}{d\mu}\right)\\
\end{aligned}
$$

#### Solving for the right term  :

$$
\begin{aligned}
\frac{d\sigma^2}{d\mu}&=\frac{d}{d\mu}\left[\frac{1}{m-1}\sum_{i=1}^{m}(x_i-\mu)^2\right]\\
&=-\frac{2}{m-1}\left[\sum_{i=1}^{m}(x_i-\mu)\right]\\
&=-\frac{2}{m-1}\left[(\sum_{i=1}^{m}x_i-\sum_{i=1}^{m}\mu)\right]\\
&=-\frac{2}{m-1}\left[(mx_i-m\mu)\right]\\
&=0
\end{aligned}
$$

#### Solving for the left term  :

$$
\begin{aligned} 
\frac{d\hat{x}}{d\mu}&=\frac{d}{d\mu}\left[\frac{x_i-\mu}{(\sigma^2+\epsilon)^\frac{1}{2}}\right]\\ 
&=(-1)(\sigma^2+\epsilon)^{-\frac{1}{2}}\\ 
&=-(\sigma^2+\epsilon)^{-\frac{1}{2}} 
\end{aligned}
$$

#### Therefore :

$$
\begin{aligned}
\frac{dL}{d\mu}&=-\sum_{i=1}^{m}\frac{d}{dy_i}
\gamma(\sigma^2+\epsilon)^{-\frac{1}{2}} +\left(\frac{dL}{d\sigma^2}\cdot 0 \right)\\
&=-\sum_{i=1}^{m}\frac{d}{dy_i}
\gamma(\sigma^2+\epsilon)^{-\frac{1}{2}}\\
\end{aligned}
$$

Code : 
```
dbndiff2 = torch.ones_like(bndiff2) * dbnvar*(1/(n-1))
dbndiff += 2*(bndiff) * dbndiff2
dbnmeani = -dbndiff.sum(0,keepdims=True)
```

#### Last node <span style="background-color:#c8facc;">(4)</span>  :

$$
\begin{aligned}
\frac{dL}{dx_i}&=
\frac{dL}{d\hat{x_i}}\frac{d\hat{x_i}}{dx_i}
+\frac{dL}{d\mu}\frac{d\mu}{dx_i}
+\frac{dL}{d\sigma^2}\frac{d\sigma^2}{dx_i}\\
\end{aligned}
$$

#### Solving terms independently  :

$$
\begin{aligned}
\frac{d\hat{x_i}}{dx_i}
&=\frac{d}{dx_i}\left[\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\right]
=(\sigma^2+\epsilon)^{-\frac{1}{2}}
\\
\frac{d\mu}{dx_i}&=
\frac{d}{dx_i}\left[\frac{1}{m}\sum_{j=1}^{m}x_j\right]
=\frac{1}{m}
\\
\frac{d\sigma^2}{dx_i}&=
\frac{d}{dx_i}\left[\frac{1}{m-1}\sum_{j=1}^{m}(x_j-\mu)^2\right]=\frac{2}{m-1}(x_i-\mu)\\
\end{aligned}
$$

#### Finally combining everything :

$$
\begin{aligned}
\frac{dL}{dx_i}&=
\frac{dL}{d\hat{x_i}}\frac{d\hat{x_i}}{dx_i}
+\frac{dL}{d\mu}\frac{d\mu}{dx_i}
+\frac{dL}{d\sigma^2}\frac{d\sigma^2}{dx_i}\\
&=\left(\frac{dL}{dy_i}\cdot\gamma\right)(\sigma^2+\epsilon)^{-\frac{1}{2}}
+\left(-\sum_{j=1}^{m}\frac{dL}{dy_j}\gamma(\sigma^2+\epsilon)^{-\frac{1}{2}}\right)\frac{1}{m}
+\left(-\frac{1}{2}\gamma\sum_{j=1}^{m}\frac{dL}{dy_j}(x_j-\mu)(\sigma^2+\epsilon)^{-\frac{3}{2}}\right)\left(\frac{2}{m-1}(x_i-\mu)\right)\\
&=\frac{\gamma(\sigma^2+\epsilon)^{-\frac{1}{2}}}{m}\left[m\frac{dL}{dy_i}-\sum_{j=1}^{m}\frac{dL}{dy_j}-\frac{m}{m-1}\hat{x}_i\sum_{j=1}^{m}\hat{x}_j\right]
\end{aligned}
$$

Overall, I just follow the same from the video leading to this code :

```
dhprebn = (bngain*bnvar_inv)/n*(n*dhpreact - dhpreact.sum(0) - n/ (n-1) * bnraw * (dhpreact*bnraw).sum(0))
```

---

Note : This repository is just collection of my notes, it does not represents how I code professionally. I'm putting it here for now as a blog.

