# 🧠 Transformer Cheatsheet

## Quick Reference (Workshop 5 of 6)

### 🛸 The Factory Analogy
A transformer is an **alien's language processing factory** with multiple stations:
- **Receiving dock** → Token embeddings
- **Stamping station** → Positional encoding  
- **Quality control teams** → Multi-head attention
- **Processing plant** → Feed-forward network
- **Assembly line** → Stacked layers

---

## Core Components

| Component | Purpose | Key Formula |
|-----------|---------|-------------|
| **Embedding** | Words → Vectors | `E[token_id]` |
| **Positional Encoding** | Add position info | `PE(pos, 2i) = sin(pos/10000^(2i/d))` |
| **Self-Attention** | Relate tokens | `softmax(QK^T/√d) × V` |
| **Multi-Head** | Multiple perspectives | Concat(head₁, ..., headₙ) × Wₒ |
| **Feed-Forward** | Transform per position | `Linear(GELU(Linear(x)))` |
| **Layer Norm** | Stabilize activations | `(x - μ) / σ` |
| **Residual** | Preserve information | `x + sublayer(x)` |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                  TRANSFORMER DECODER                 │
├─────────────────────────────────────────────────────┤
│  Input: "the cat sat"                               │
│         ↓                                           │
│  ┌─────────────────────────────────────────────┐   │
│  │  Token Embedding + Positional Encoding       │   │
│  └─────────────────────────────────────────────┘   │
│         ↓                                           │
│  ┌─────────────────────────────────────────────┐   │
│  │  Transformer Block × N                       │   │
│  │  ┌─────────────────────────────────────┐    │   │
│  │  │  Multi-Head Self-Attention (masked)  │    │   │
│  │  │  + Residual & LayerNorm              │    │   │
│  │  └─────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────┐    │   │
│  │  │  Feed-Forward Network                │    │   │
│  │  │  + Residual & LayerNorm              │    │   │
│  │  └─────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────┘   │
│         ↓                                           │
│  ┌─────────────────────────────────────────────┐   │
│  │  Output Projection → Vocabulary Logits       │   │
│  └─────────────────────────────────────────────┘   │
│         ↓                                           │
│  Output: probability distribution over next token  │
└─────────────────────────────────────────────────────┘
```

---

## Code Snippets

### Initialize a Mini Transformer
```python
from transformer import MiniTransformer, SimpleVocab

vocab = SimpleVocab()
model = MiniTransformer(
    vocab_size=vocab.vocab_size,
    embed_dim=64,
    num_heads=4,
    num_layers=2
)
```

### Generate Text
```python
prompt = "the cat sat"
token_ids = vocab.encode(prompt)
generated = model.generate(token_ids, max_new_tokens=5)
output = vocab.decode(generated)
print(output)  # "the cat sat on the mat"
```

### Inspect Attention
```python
token_ids = vocab.encode("the cat sat")
logits, all_weights = model.forward(token_ids)

# all_weights: list of (num_heads, seq_len, seq_len) per layer
layer_0_weights = all_weights[0]  # First layer
head_0_weights = layer_0_weights[0]  # First head
```

---

## Key Hyperparameters

| Parameter | Typical Values | Our Demo | GPT-3 |
|-----------|----------------|----------|-------|
| embed_dim | 64-12288 | 64 | 12288 |
| num_heads | 4-96 | 4 | 96 |
| num_layers | 2-96 | 2 | 96 |
| hidden_dim | 4×embed_dim | 256 | 49152 |
| vocab_size | 100-50000 | ~30 | 50257 |

---

## Attention Types

| Type | Can See | Use Case |
|------|---------|----------|
| **Bidirectional** | All tokens | BERT (understanding) |
| **Causal (Masked)** | Past + current only | GPT (generation) |
| **Cross-Attention** | Encoder outputs | Translation, T5 |

---

## Common Gotchas

⚠️ **Causal Mask**: Decoder must not see future tokens!
```python
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
```

⚠️ **Scale Factor**: Divide by √d to prevent softmax saturation
```python
attention = softmax(QK^T / sqrt(head_dim))
```

⚠️ **Position Matters**: Without positional encoding, "dog bites man" = "man bites dog"

---

## Remember 💡

> **A transformer is just attention + feed-forward, stacked N times, 
> with residual connections and layer normalization.**
> 
> The magic isn't in any single component—it's in how they work together 
> to build increasingly abstract representations of language.
