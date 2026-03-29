# Research Foundations

Papers this work builds on and their key contributions.

## Core: Modern Hopfield Networks

**"Hopfield Networks is All You Need"**
Ramsauer, Schäfl, Lehner, Seidl, Widrich, Adler, Gruber, Holzleitner, Pavlovic, Sandve, Unterthiner, Hochreiter
ICLR 2021 | https://arxiv.org/abs/2008.02217

- Reformulated classical Hopfield with continuous states and exponential energy
- Storage capacity: exponential in dimension (vs linear for classical)
- Update rule: ξ_new = X · softmax(β · X^T · ξ) — equivalent to transformer attention
- Official library: https://github.com/ml-jku/hopfield-layers

## Sparse Hopfield Networks

**"Sparse and Structured Hopfield Networks"**
Santos, Niculae, McAuley, Martins
ICML 2024 | https://arxiv.org/abs/2402.13725

- Replace softmax with sparse transformations (entmax, sparsemax) in Hopfield energy
- Produces exact zeros on irrelevant patterns (not tiny-but-nonzero weights)
- SparseMAP variant retrieves pattern ASSOCIATIONS (document clusters) instead of single patterns
- Connection between loss margins, sparsity, and exact memory retrieval
- Tested on multiple instance learning and text rationalization

Key relevance: sparse attention makes meta-stable states interpretable — when
the sparse Hopfield converges between k documents, those k documents get non-zero
weight and everything else is exactly 0. This is structured cluster retrieval.

## Hopfield Encoding Networks (HEN)

**"Hopfield Encoding Networks"**
Kashyap et al., 2024

- Encodes documents into a learned latent space before Hopfield storage
- The encoder pushes similar-but-different documents apart, preventing meta-stable states
- Addresses the core problem: when stored patterns are too similar, the network
  converges to a blend rather than a single pattern

## Continuous-Time Hopfield

**"Modern Hopfield Networks with Continuous-Time Memories"**
Santos, Martins et al., 2025

- Retrieval modeled as a continuous ODE instead of discrete update steps
- Adaptive computation: hard queries get more steps, easy queries converge fast
- Stronger convergence guarantees via Lyapunov stability theory

## Hopfield in Transformer Architectures

**"Routing without Forgetting"**
Masano, Bellitto, Goswani, Van de Weijer, Spampinato
March 2026 | https://arxiv.org/abs/2603.09576

- Augments transformer layers with energy-based associative retrieval (Modern Hopfield)
- Generates dynamic prompts through single-step associative retrieval at each layer
- Uses closed-form energy minimization (no iterative steps)
- Applied to continual learning, not LLM memory — but the architecture pattern is identical

## Our Prior Work

**"Improving Out-of-Distribution Data Handling and Corruption Resistance via Modern Hopfield Networks"**
Sargolzaei, Rueda
ICPR 2024 | https://arxiv.org/abs/2408.11309

- HopfieldPooling trained as a denoising autoencoder (MSE loss)
- Integrated into frozen classifier at test time
- 13.84% improvement in corruption accuracy on MNIST-C
- Demonstrated that Hopfield associative memory can recover clean patterns from corrupted inputs

## What We're Building

**Sparse Hopfield memory layers injected into a frozen LLM for document retrieval.**

Novel contributions:
1. Sparse Hopfield (entmax) inside LLM hidden layers for memory injection — not done before
   (Santos et al. tested on MIL/rationalization, not LLM memory)
2. Meta-stable states as group knowledge: when sparse attention spreads across k documents,
   interpret this as cluster retrieval rather than retrieval failure
3. Multi-layer injection with different sparsity patterns at different depths:
   - Early layers: broader attention → topic-level retrieval
   - Late layers: sharper attention → fact-level retrieval
4. Evaluation framework: perplexity under correct/half/random memory conditions
