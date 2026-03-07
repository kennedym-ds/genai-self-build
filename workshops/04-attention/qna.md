# ❓ Workshop 4: Attention Mechanism - Q&A

## Conceptual Questions

### Q: Why do we need attention? What was wrong with the old approaches?

**🛸 Analogy:** Imagine our alien is translating a long document. Without attention, they'd have to memorize the ENTIRE document into a fixed-size summary before translating. Important details get lost!

**Technical Answer:** Before attention (2014), sequence models used fixed-size hidden states to compress entire sequences. This created a "bottleneck" where:
1. Long sequences lost information
2. Early words were forgotten
3. The model couldn't "look back" at specific positions

Attention solved this by letting the model directly access any position at any time—no compression needed!

---

### Q: What exactly are Query, Key, and Value?

**🛸 Analogy:** Think of a library:
- **Query (Q):** Your search term ("books about cats")
- **Key (K):** The title/description of each book
- **Value (V):** The actual book content

You match your Query against all Keys, then retrieve the Values of matching books (weighted by how good the match was).

**Technical Answer:**
- **Query:** "What am I looking for?" - derived from the current position
- **Key:** "What do I contain?" - derived from each position being attended to
- **Value:** "What information do I provide?" - the actual content to retrieve

All three are linear projections of the input: `Q = X @ W_q`, `K = X @ W_k`, `V = X @ W_v`

---

### Q: Why do we scale by √d? Can we skip it?

**🛸 Analogy:** Imagine our spotlight has a brightness knob. Without scaling, the knob would be WAY too sensitive—tiny differences would make the spotlight go from completely off to blinding. Scaling makes the knob work smoothly.

**Technical Answer:** The dot product of two d-dimensional vectors has variance proportional to d. For d=64:
- Without scaling: scores range roughly -64 to +64
- Softmax of these → nearly one-hot (0.999 for max, ~0 for others)
- Gradients become very small → training stalls

With scaling by √d:
- Scores range roughly -8 to +8
- Softmax → smooth distribution
- Gradients flow properly

**Can you skip it?** Technically yes, but training will be much harder and you may need smaller learning rates.

---

### Q: What's the difference between self-attention and cross-attention?

**🛸 Analogy:** 
- **Self-attention:** Our alien reading a book and connecting different parts of the SAME book
- **Cross-attention:** Our alien reading a book while referencing a DIFFERENT book (like original text vs translation)

**Technical Answer:**
- **Self-attention:** Q, K, V all come from the same sequence
  - `Q = X @ W_q`, `K = X @ W_k`, `V = X @ W_v` (same X)
  - Used in: GPT, BERT encoder
  
- **Cross-attention:** Q comes from one sequence, K/V from another
  - `Q = X @ W_q`, `K = Y @ W_k`, `V = Y @ W_v` (different X and Y)
  - Used in: Encoder-decoder models, image-to-text

---

## Technical Questions

### Q: Why do we need multiple heads? Isn't one enough?

**🛸 Analogy:** One spotlight can only focus on one thing at a time. But language has MANY relationships happening simultaneously! Multiple spotlights let us track grammar, meaning, pronouns, and position all at once.

**Technical Answer:** A single attention head can only capture one type of relationship. Multi-head attention:
1. Splits the embedding into `h` parts (e.g., 64 → 4 × 16)
2. Each head learns different relationships
3. Heads are concatenated and projected back

Research has shown heads specialize:
- Some track syntax (subject-verb)
- Some track semantics (synonyms)
- Some track position (nearby words)
- Some track coreference (pronouns)

---

### Q: How does causal masking work exactly?

**🛸 Analogy:** Our alien is reading a mystery novel but MUST cover up everything after the current page. They can only use clues from pages they've already read. This prevents "cheating" by looking at the answer!

**Technical Answer:** We add a mask to attention scores before softmax:

```python
# Causal mask (upper triangle = -inf)
mask = [[0, -∞, -∞, -∞],
        [0,  0, -∞, -∞],
        [0,  0,  0, -∞],
        [0,  0,  0,  0]]

scores = Q @ K.T + mask  # Add mask
weights = softmax(scores)  # -∞ → 0 after softmax
```

After softmax, positions with -∞ become 0, so future positions get zero attention.

---

### Q: What's the computational complexity of attention?

**🛸 Analogy:** If our alien wants every performer to look at every other performer, that's n² pairs to consider! Double the performers = quadruple the work.

**Technical Answer:** Self-attention is O(n² · d) where:
- n = sequence length
- d = embedding dimension

This is the main bottleneck for long sequences. A 10,000 token sequence needs 100 million attention computations!

**Solutions being researched:**
- Sparse attention (only attend to nearby + selected positions)
- Linear attention (approximate with O(n) complexity)
- Flash Attention (GPU-optimized to reduce memory)

---

## Real-World Questions

### Q: How does ChatGPT use attention?

**Technical Answer:** GPT uses **causal multi-head self-attention** in every layer:

```
GPT-3/4 Architecture:
├── 96 transformer layers
│   └── Each has multi-head attention
│       ├── 96 attention heads
│       ├── 128 dimensions per head
│       └── Causal masking (can't see future)
└── 12,288 total embedding dimension
```

Every token attends to all previous tokens at every layer. For a 4,096 token context:
- 96 layers × 4,096² attention pairs = ~1.6 billion attention computations per forward pass!

---

### Q: How does BERT's attention differ from GPT's?

**Technical Answer:**

| Aspect | BERT | GPT |
|--------|------|-----|
| **Masking** | Bidirectional (see all) | Causal (see past only) |
| **Training** | Masked language modeling | Next token prediction |
| **Use case** | Understanding (classify, QA) | Generation (chat, code) |
| **Attention pattern** | Full square matrix | Lower triangular |

BERT: "Fill in the [MASK]" (needs context from both sides)
GPT: "What comes next?" (only uses left context)

---

### Q: What are attention heads actually learning?

**Research Finding:** Studies like "What do attention heads do?" have found:
- **Positional heads:** Attend to previous/next token
- **Syntactic heads:** Track grammatical structure
- **Semantic heads:** Find related concepts
- **Rare token heads:** Flag unusual words
- **Copy heads:** Repeat earlier tokens

You can visualize this in tools like BertViz or our Streamlit app!

---

## Common Misconceptions

### Q: Does higher attention weight mean the word is more "important"?

**Not exactly!** Attention weights show what's relevant FOR A SPECIFIC QUERY. A word might have:
- High weight when queried by pronouns (referent)
- Low weight when queried by other words
- Different weights from different heads

Think of it as: "How relevant is this for understanding THIS position?"

---

### Q: Is attention the same as memory?

**No!** Attention is more like "looking up" than "remembering":
- Memory = stored representations that persist
- Attention = dynamic routing that happens each forward pass

The model recomputes attention every time. It doesn't "remember" previous attention patterns.

---

### Q: Can attention replace everything else in a neural network?

**Almost!** The Transformer architecture uses attention + two other key components:
1. **Feed-forward networks:** Process each position independently
2. **Layer normalization:** Stabilize training

The paper "Attention Is All You Need" removed RNNs/CNNs but kept these.

---

## Debugging Questions

### Q: My attention weights are all nearly equal (uniform). Why?

**Possible causes:**
1. **Not enough training:** Weights haven't learned to differentiate
2. **Scaling issue:** Try adjusting the scaling factor
3. **Initialization:** Random weights give random attention
4. **No meaningful relationships:** Random input → random attention

In our demo, we use random embeddings, so attention is somewhat random. In a trained model, you'd see clear patterns.

---

### Q: Why are my attention weights NaN or Inf?

**Possible causes:**
1. **Missing scaling:** Huge dot products → softmax overflow
2. **Mask with wrong values:** Use -1e9 instead of actual -inf
3. **Numerical issues:** Check for zeros in denominators

**Fix:** Ensure you're using `scaled_dot_product_attention` and that masks use `-1e9` (not `float('-inf')`).

---

<div align="center">

**🎯 Workshop 4 of 6** | *GenAI Self-Build Workshop Series*

</div>
