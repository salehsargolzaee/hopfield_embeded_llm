# Sentence Embeddings

## What they are

A sentence embedding is a fixed-length vector (list of numbers) that represents the meaning of a text. The key property: texts with similar meanings get vectors that are close together in the vector space.

```
"How do I reset my password?"  → [0.12, -0.34, 0.56, ..., 0.78]   (384 numbers)
"I forgot my login credentials" → [0.11, -0.31, 0.54, ..., 0.80]  (close to above!)
"The weather is nice today"    → [-0.45, 0.22, -0.18, ..., 0.03]  (far away)
```

## How the model is trained

Models like `all-MiniLM-L6-v2` start with a pre-trained transformer (like BERT) and fine-tune it on pairs of sentences:

1. **Positive pairs**: Sentences that should be similar (paraphrases, question-answer pairs)
2. **Negative pairs**: Sentences that should be different (random pairs from different topics)

The training objective pushes positive pairs close together and negative pairs far apart in the embedding space. After training, the model generalizes — it produces meaningful embeddings for text it has never seen.

## The model we use: all-MiniLM-L6-v2

| Property | Value |
|----------|-------|
| Output dimension | 384 |
| Model size | ~80 MB |
| Speed | Fast (small model) |
| Quality | Good general-purpose quality |
| Max input length | 256 tokens |

This is a good default for prototyping. For higher quality at the cost of speed, you could swap to `all-mpnet-base-v2` (768 dimensions, ~400 MB). This is a config change — no code modification needed.

## Why normalization matters

We L2-normalize all embeddings to unit length (||v|| = 1). This means:

```
cosine_similarity(a, b) = dot_product(a, b)
```

When vectors have unit length, the dot product equals the cosine of the angle between them. This simplifies the math in both the cosine retriever and the Hopfield retriever — a single matrix multiply computes all similarities at once.

## Limitations

- **Out-of-vocabulary tokens**: The model has a fixed vocabulary. Rare tokens (product codes, serial numbers, internal acronyms) may get split into subwords or mapped to [UNK], losing their meaning. This is where BM25's exact matching has an edge.
- **Max sequence length**: Text beyond 256 tokens gets truncated. This is why chunking matters — we split documents into pieces that fit within this limit.
- **Static embeddings**: The embedding of a chunk is computed once and cached. If the chunk's text changes, you need to re-embed.
- **Domain shift**: The model was trained on general-purpose text. Highly technical or domain-specific language (legal, medical, internal jargon) may not embed as well. Fine-tuning the embedding model on domain data is a possible future improvement.

## In this project

The `Embedder` class in `src/embedding/embedder.py` wraps the sentence-transformer model. It:

1. Loads the model once at startup
2. Embeds chunks in batches (configurable batch size)
3. Caches results to disk (hash-based naming so different data → different cache files)
4. Provides `embed_single()` for individual queries during retrieval

Both the Hopfield retriever and cosine retriever consume these embeddings. BM25 does not — it works on raw text.
