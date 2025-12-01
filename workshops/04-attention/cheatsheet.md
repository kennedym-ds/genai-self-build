# 📋 Attention Mechanism Cheatsheet

## 🛸 The One-Liner
> **Attention = Spotlight that focuses on relevant words based on what we're looking for**

---

## 🔑 Core Formula

```
Attention(Q, K, V) = softmax(Q @ K.T / √d) @ V
```

| Symbol | Name | What It Is |
|--------|------|------------|
| Q | Query | What we're looking for |
| K | Key | Labels for each position |
| V | Value | Content at each position |
| d | Dimension | Size of Q/K vectors |
| √d | Scale | Prevents extreme softmax |

---

## 📊 Quick Code Reference

```python
from attention import SimpleAttention
import numpy as np

# Initialize
attn = SimpleAttention(embed_dim=64, num_heads=4)

# Input: (seq_len, embed_dim)
x = np.random.randn(10, 64).astype(np.float32)

# Self-attention (bidirectional)
output = attn.self_attention(x)

# Causal attention (GPT-style)
output = attn.self_attention(x, causal=True)

# With weights
output, weights = attn.forward(x, return_weights=True)
# weights shape: (num_heads, seq_len, seq_len)
```

---

## 🎭 Attention Types

| Type | Formula | Use Case |
|------|---------|----------|
| **Dot-Product** | Q @ K.T | Simple, fast |
| **Scaled** | (Q @ K.T) / √d | Standard in transformers |
| **Multi-Head** | Concat(head₁...headₙ) | Multiple relationship types |
| **Causal** | + causal mask | Text generation (GPT) |

---

## 🔢 Shape Reference

```
Input:   (seq_len, embed_dim)     e.g., (10, 64)
Q, K, V: (seq_len, embed_dim)     e.g., (10, 64)
Scores:  (seq_len, seq_len)       e.g., (10, 10)
Weights: (seq_len, seq_len)       e.g., (10, 10)
Output:  (seq_len, embed_dim)     e.g., (10, 64)

Multi-Head:
Weights: (num_heads, seq_len, seq_len)  e.g., (4, 10, 10)
```

---

## 🎭 Causal Mask

```python
# Create causal mask
def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask[mask == 1] = -np.inf
    return mask

# Visual:
# [0, -∞, -∞, -∞]   Position 0 sees: itself only
# [0,  0, -∞, -∞]   Position 1 sees: 0, 1
# [0,  0,  0, -∞]   Position 2 sees: 0, 1, 2
# [0,  0,  0,  0]   Position 3 sees: all
```

---

## 🧠 Multi-Head Intuition

| Head | Might Learn |
|------|-------------|
| Head 1 | Subject-verb agreement |
| Head 2 | Pronoun resolution |
| Head 3 | Adjacent word relationships |
| Head 4 | Long-range dependencies |

```python
# Split embedding across heads
head_dim = embed_dim // num_heads  # 64 / 4 = 16
```

---

## ⚡ Why Scale by √d?

```
Without scaling (d=64):
  score = Q @ K.T ≈ sum of 64 products
  → variance grows with d
  → softmax becomes nearly one-hot
  → gradients vanish!

With scaling:
  score = (Q @ K.T) / √64 = (Q @ K.T) / 8
  → variance stays ~1
  → softmax stays smooth
  → gradients flow!
```

---

## 🆚 BERT vs GPT

| | BERT | GPT |
|--|------|-----|
| **Masking** | Bidirectional | Causal |
| **Sees** | All positions | Only past |
| **Use** | Understanding | Generation |
| **Example** | Sentiment analysis | ChatGPT |

---

## 💡 Remember

> **Attention weights always sum to 1** (softmax output)
> 
> Each row in the attention matrix is a probability distribution over positions.
> 
> High weight = "pay attention to this word"
> Low weight = "ignore this word"

---

## 🔗 Quick Links

- Workshop 4 files: `workshops/04-attention/`
- Run demo: `streamlit run app.py`
- Run tests: `python test_attention.py`

---

<div align="center">

**🎯 Workshop 4 of 6** | *GenAI Self-Build Series*

</div>
