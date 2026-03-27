# Semantic Chunking

## The problem

Documents are too long to embed directly. Embedding models have a max input length (256 tokens for MiniLM), and even if they didn't, a single vector for a 50-page document would be too coarse to answer specific questions.

So we split documents into chunks. But *how* you split them matters a lot.

## Fixed-window chunking (the bad way)

Split every N tokens, regardless of content:

```
Original: "TCP uses a three-way handshake. The client sends a SYN packet.
           The server responds with SYN-ACK. The client then sends ACK."

Fixed chunks (max 8 tokens):
  Chunk 1: "TCP uses a three-way handshake. The client"
  Chunk 2: "sends a SYN packet. The server responds with"
  Chunk 3: "SYN-ACK. The client then sends ACK."
```

Chunk 1 ends mid-sentence. Its embedding represents a fragment, not a complete thought. If someone asks "How does the TCP handshake work?", the embedding of "TCP uses a three-way handshake. The client" is a poor representation of the full answer.

## Semantic chunking (what we do)

Split on sentence boundaries, grouping sentences until we approach the max size:

```
Semantic chunks (max 15 tokens):
  Chunk 1: "TCP uses a three-way handshake. The client sends a SYN packet."
  Chunk 2: "The server responds with SYN-ACK. The client then sends ACK."
```

Each chunk contains complete sentences. The embedding of chunk 1 captures "TCP handshake, client SYN" as a coherent idea.

## The overlap

Consecutive chunks share some trailing/leading sentences:

```
Without overlap:
  Chunk 1: [Sentences 1, 2, 3]
  Chunk 2: [Sentences 4, 5, 6]
  → If the answer spans sentences 3-4, neither chunk has the full picture.

With overlap:
  Chunk 1: [Sentences 1, 2, 3]
  Chunk 2: [Sentences 3, 4, 5]
  → Sentence 3 appears in both chunks. At least one chunk has context from both sides.
```

The `overlap_tokens` config parameter controls how much overlap. Too much overlap wastes storage (duplicate content). Too little risks splitting critical context. 32 tokens (roughly 1-2 sentences) is a reasonable starting point.

## Our implementation

The chunker in `src/ingestion/chunker.py`:

1. Uses NLTK's `sent_tokenize()` to split text into sentences
2. Groups sentences into chunks up to `max_chunk_tokens` (default 256)
3. When a chunk fills up, starts a new chunk with the last few sentences carried over (overlap)
4. Drops chunks shorter than `min_chunk_tokens` (default 20) — these are usually noise (page numbers, headers, etc.)

## Why this matters for retrieval

The quality of your embeddings depends directly on the quality of your chunks. A clean, coherent chunk produces a meaningful embedding that's easy to match against queries. A fragmented chunk produces a noisy embedding that could match unrelated queries.

This is especially relevant for the Hopfield retriever. The energy landscape has one attractor per stored pattern (chunk embedding). If the embeddings are noisy, the attractors are poorly separated, leading to more meta-stable states (false retrievals). Clean chunks → clean embeddings → well-separated attractors → better retrieval.

## Configuration

```yaml
chunking:
  strategy: "semantic"    # or "fixed" for the naive approach
  max_chunk_tokens: 256   # max words per chunk
  overlap_tokens: 32      # words shared between consecutive chunks
  min_chunk_tokens: 20    # drop chunks smaller than this
```
