<p align="center">
  <img src="https://raw.githubusercontent.com/Deepity-HQ/Deepity/main/resources/deepity-mark.png" alt="Deepity logo" width="200"/>
</p>

<h1 align="center">Deepity</h1>

<p align="center">
  <b>A fast C++ / CUDA engine for training neural networks with predictive coding, a brain-inspired alternative to backpropagation with a simple Python interface.</b>
</p>

<p align="center">
  <a href="https://ra4ster.github.io/Deepity">Docs</a> •
  <a href="https://github.com/Deepity-HQ/Deepity">Source</a> •
  <a href="https://pypi.org/project/pydeepity/">PyPI</a> •
  <a href="https://github.com/Deepity-HQ/Deepity#contributing">Contributing</a>
</p>

---

### What we're building

Most neural networks learn with **backpropagation**: one global error signal flows backward through every layer. **Predictive coding networks (PCNs)** learn differently — each layer only ever reacts to the mismatch between its own activity and a prediction from the layer next to it, a theory with roots in neuroscience for how the brain itself learns.

**Deepity** is a from-scratch engine for training PCNs quickly: a hand-tuned C++20 core (AVX2/AVX-512 SIMD, OpenBLAS/MKL) with an active CUDA backend, and a Python package (`pydeepity`) on top so you can define and train a network in a few lines.

```bash
pip install pydeepity
```

```python
from pydeepity import SimplePCN, Linear, TanH

net = SimplePCN(Linear(2, 8), TanH(), Linear(8, 1), batch_size=1)
net.configure(learning_rate=0.05, inference_rate=0.3, optimizer="SGD")
```

On MNIST, Deepity's `SimplePCN` reaches **98.21%** test accuracy in **487.5s** on a laptop CPU — faster than the two established predictive-coding libraries it was benchmarked against (JAX-based `jpc`, and `ngc-learn`) at matching or better accuracy. See the [full benchmark writeup](https://github.com/Deepity-HQ/Deepity#performance--benchmarks) for the exact setup and where Deepity does *not* win.

### Where we are

Deepity started as an academic research project and is now becoming infrastructure other predictive-coding research can build on. The CPU path is the mature one; the CUDA backend is under active development, with large-scale ImageNet training as the current focus.

We're looking for contributors with **C++/CUDA performance** or **Python build/DevOps** experience: see the main repo's open issues.

---

<p align="center"><sub>MIT License</sub></p>
