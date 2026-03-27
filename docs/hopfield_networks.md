# Modern Hopfield Networks

## Classical Hopfield Networks (1982)

John Hopfield introduced a model of associative memory: a network that stores patterns and can recall the full pattern when given a noisy or partial version.

Think of it like autocomplete for memory. You store the pattern "HELLO" and then give the network "H_LLO" — it fills in the gap and returns "HELLO."

### How classical Hopfield works

The network has N binary neurons (values +1 or -1). You store P patterns by constructing a weight matrix:

```
W = (1/N) Σ ξᵘ (ξᵘ)ᵀ
```

Where ξᵘ is the u-th stored pattern. To recall, you start with a probe state and repeatedly update each neuron:

```
sᵢ = sign(Σⱼ Wᵢⱼ sⱼ)
```

The network converges to the nearest stored pattern (energy minimum).

### The problems

1. **Low capacity**: You can store at most ~0.14N patterns reliably. Beyond that, patterns interfere with each other.
2. **Spurious states**: The network sometimes converges to a "ghost" pattern — a point in state space that wasn't stored but is a stable fixed point of the dynamics. These are false memories.
3. **Binary only**: Patterns must be binary vectors. Real-world data (like text embeddings) is continuous.

## Modern Hopfield Networks (Ramsauer et al., 2020)

The 2020 paper "Hopfield Networks is All You Need" replaced the classical energy function with a new one that works on continuous vectors and has exponential storage capacity.

### The energy function

```
E(ξ) = -lse(β, X^T ξ) + (1/2) ξ^T ξ + const
```

Where:
- **ξ** is the state vector (your query, e.g., a question embedding)
- **X** is a matrix whose columns are stored patterns (document chunk embeddings)
- **β** is the inverse temperature parameter
- **lse** is the log-sum-exp function: `lse(β, z) = (1/β) log(Σᵢ exp(β zᵢ))`

The log-sum-exp is a smooth approximation of the max function. As β increases, lse approaches max. This matters because it controls how "sharp" the energy landscape is.

### The update rule (retrieval)

To find the stored pattern closest to query ξ, apply:

```
ξ_new = X · softmax(β · X^T · ξ)
```

Breaking this down step by step:

1. `X^T · ξ` — compute similarity of the query to every stored pattern. This gives an N-dimensional vector of scores.
2. `β · (...)` — scale by inverse temperature. High β amplifies differences between scores.
3. `softmax(...)` — convert scores to probabilities. The highest-scoring pattern gets the most weight.
4. `X · (...)` — take a weighted sum of all stored patterns using those probabilities.

The result is a new state vector that's "closer" to the most relevant stored pattern.

### Why this is the same as attention

If you squint at the update rule, it's exactly the attention mechanism from transformers:

```
Attention(Q, K, V) = softmax(Q K^T / √d) V
```

The Hopfield update is:

```
ξ_new = X · softmax(β · X^T · ξ)
```

Map: Q = ξ (query), K = V = X (stored patterns), β = 1/√d (temperature scaling).

This isn't a coincidence. Ramsauer et al. showed that transformer attention is a single step of Modern Hopfield retrieval. Our retriever makes this connection explicit.

### Storage capacity

Classical: O(N) patterns for N neurons.
Modern: **exponential** in the dimension d. Specifically, you can store up to `exp(d/2)` patterns before interference becomes a problem.

For 384-dimensional embeddings (MiniLM-L6-v2), that's exp(192) ≈ 10^83 patterns. In practice, we'll never hit this limit.

### Multi-step retrieval

With `num_steps=1`, the retriever does a single attention operation (like a transformer). With `num_steps>1`, the state vector iterates:

```
ξ₀ = query embedding
ξ₁ = X · softmax(β · X^T · ξ₀)
ξ₂ = X · softmax(β · X^T · ξ₁)
...
```

Each step moves ξ closer to the nearest energy minimum. This can help with ambiguous queries that sit roughly equidistant from multiple patterns — the iterative dynamics "commit" to one attractor instead of averaging across several.

Whether multi-step actually improves retrieval in practice is an empirical question that our evaluation harness tests.

## How we use it in this project

Document chunk embeddings are stored as rows of the pattern matrix X. When a query comes in:

1. Embed the query using the same sentence-transformer
2. Run the Hopfield update rule for `num_steps` iterations
3. Score each stored pattern by its final attention weight
4. Return the top-k highest-weighted patterns as results

The scores are softmax attention weights, so they sum to 1 and can be interpreted as "how much this chunk contributed to the retrieval."

## The β parameter

β is the single most important hyperparameter. It controls the sharpness of the softmax:

- **β = 1**: Flat softmax. Many patterns get similar weights. The retrieval is "blurry" — you get a mix of related content.
- **β = 10**: Moderate sharpness. A few patterns dominate.
- **β = 100**: Nearly argmax. One pattern gets almost all the weight. The retrieval is decisive but may be too rigid.
- **β → ∞**: Pure argmax. Equivalent to cosine similarity nearest-neighbor search.

The optimal β depends on the data. If your document chunks are very diverse (different topics), a high β works well. If chunks are similar to each other (same-domain enterprise docs), a lower β might help by blending related context.

## References

- Ramsauer, H., et al. (2020). "Hopfield Networks is All You Need." arXiv:2008.02217
- Original: Hopfield, J.J. (1982). "Neural networks and physical systems with emergent collective computational abilities." PNAS.
