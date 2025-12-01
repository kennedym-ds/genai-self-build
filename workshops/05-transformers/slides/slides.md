---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #ffffff
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
  }
  h1 {
    color: #667eea;
  }
  h2 {
    color: #f093fb;
  }
  code {
    background: #2d2d44;
    border-radius: 4px;
    padding: 2px 8px;
  }
  pre {
    background: #2d2d44;
    border-radius: 8px;
    padding: 16px;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .highlight {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
  }
  table {
    font-size: 0.85em;
  }
---

# 🧠 Workshop 5: Transformers
## The Alien's Complete Brain

*Building the architecture that powers ChatGPT, Claude, and LLaMA*

**Workshop 5 of 6** | GenAI Self-Build Series

---

# 📋 Learning Objectives

By the end of this workshop, you will:

1. **Understand** the complete transformer architecture
2. **Build** a minimal decoder-only transformer (GPT-style)
3. **Visualize** how attention, FFN, and normalization work together
4. **Generate** text token by token

---

# 🛸 The Factory Analogy

Our alien has evolved from a spotlight to a **complete language processing factory**

```
┌──────────────────────────────────────────────────────────┐
│                    ALIEN'S BRAIN FACTORY                  │
├──────────────────────────────────────────────────────────┤
│  📦 Receiving     Words → Number vectors                 │
│  📍 Stamping      Add position markers                   │
│  👀 Quality Ctrl  Multiple inspection teams              │
│  🔧 Processing    Transform & refine                     │
│  🔄 Repeat        Multiple assembly line passes          │
│  📤 Shipping      Output next word prediction            │
└──────────────────────────────────────────────────────────┘
```

---

# 🏗️ Transformer Architecture Overview

```
Input: "the cat sat"
         ↓
┌─────────────────────────────────────┐
│     Token Embedding + Position       │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Transformer Block × N            │
│  ┌─────────────────────────────┐    │
│  │  Multi-Head Self-Attention   │    │
│  │  + Add & LayerNorm           │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │  Feed-Forward Network        │    │
│  │  + Add & LayerNorm           │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│    Output: Next token probabilities  │
└─────────────────────────────────────┘
```

---

# 🔢 Step 1: Token Embedding

Convert words to vectors (we did this in Workshop 2!)

```python
# Each word becomes a vector of numbers
embedding = np.random.randn(vocab_size, embed_dim)

# "cat" → embedding[cat_id] → [0.2, -0.5, 0.8, ...]
```

**Why?** Neural networks work with numbers, not words

---

# 📍 Step 2: Positional Encoding

**Problem:** Transformers process all tokens in parallel — no inherent order!

```python
# Without position: "dog bites man" = "man bites dog" 😱
# With position: Each word gets a unique position stamp
```

**Solution:** Sine and cosine waves at different frequencies

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d))  # Even dimensions
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))  # Odd dimensions
```

---

# 📍 Positional Encoding Visualization

```
Position 0: [sin(0), cos(0), sin(0), cos(0), ...]
Position 1: [sin(1), cos(1), sin(0.01), cos(0.01), ...]
Position 2: [sin(2), cos(2), sin(0.02), cos(0.02), ...]
...
```

**Key insight:** 
- Low dimensions → High frequency (vary quickly between positions)
- High dimensions → Low frequency (capture longer-range patterns)

Every position gets a **unique fingerprint**!

---

# 👀 Step 3: Multi-Head Attention

We covered attention in Workshop 4. Now we add **multiple heads**!

```python
# Instead of one attention mechanism:
attention_1 = compute_attention(Q1, K1, V1)  # Syntax patterns
attention_2 = compute_attention(Q2, K2, V2)  # Coreference
attention_3 = compute_attention(Q3, K3, V3)  # Proximity
attention_4 = compute_attention(Q4, K4, V4)  # Semantics

# Combine all perspectives
output = concat(attention_1, ..., attention_4) @ W_out
```

Each head can learn **different types of relationships**!

---

# 🎭 Multi-Head: Different Perspectives

| Head | Might Learn To Focus On |
|------|------------------------|
| Head 1 | Subject-verb agreement ("The **cats** **are**...") |
| Head 2 | Pronoun resolution ("John said **he**...") |
| Head 3 | Adjacent words (local context) |
| Head 4 | Long-range dependencies |

**Analogy:** Multiple quality control teams, each specialized!

---

# 🎭 Causal Masking (Decoder)

In GPT-style models, we **can't look at future tokens**:

```
          the   cat   sat   on
the       ✅    ❌    ❌    ❌
cat       ✅    ✅    ❌    ❌
sat       ✅    ✅    ✅    ❌
on        ✅    ✅    ✅    ✅
```

**Why?** During training, we predict next tokens — can't cheat by looking ahead!

---

# 🔧 Step 4: Feed-Forward Network

After attention mixes information, FFN processes each position:

```python
def feed_forward(x):
    # Expand: embed_dim → 4 × embed_dim
    hidden = linear1(x)
    
    # Non-linearity (GELU)
    hidden = gelu(hidden)
    
    # Compress: 4 × embed_dim → embed_dim  
    output = linear2(hidden)
    
    return output
```

**Key:** Applied to each position **independently**

---

# 🔧 Why GELU Activation?

```
GELU vs ReLU:

ReLU:  ────────╱          Hard cutoff at 0
              ╱
             ╱

GELU:  ──────╭────        Smooth curve
           ╭╯
          ╯
```

**GELU advantages:**
- Smoother gradients
- Allows small negative values
- Better empirical performance

---

# 🔄 Step 5: Residual Connections

**The secret sauce for deep networks!**

```python
# Instead of:
x = sublayer(x)

# We do:
x = x + sublayer(x)  # Add input back!
```

**Benefits:**
1. Gradients flow directly backward
2. Original information preserved
3. Network learns **refinements**, not replacements

Without residuals, 96-layer transformers wouldn't train!

---

# 📊 Step 6: Layer Normalization

Normalize activations to stable range:

```python
def layer_norm(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)
```

**Why?** Prevents activations from exploding or vanishing

**Where?** After each sublayer (attention, FFN)

---

# 🏭 Complete Transformer Block

```python
class TransformerBlock:
    def forward(self, x):
        # 1. Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # 2. Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

**Stack N of these** for a complete transformer!

---

# 🆚 Transformer Variants

| Type | Attention | Use Case | Examples |
|------|-----------|----------|----------|
| **Encoder-only** | Bidirectional | Understanding | BERT, RoBERTa |
| **Decoder-only** | Causal | Generation | GPT, Claude, LLaMA |
| **Encoder-Decoder** | Both | Translation | T5, BART |

**Today:** We build a **decoder-only** model (like GPT)

---

# 💻 Live Demo: Building MiniTransformer

```python
from transformer import MiniTransformer, SimpleVocab

# Initialize
vocab = SimpleVocab()
model = MiniTransformer(
    vocab_size=vocab.vocab_size,
    embed_dim=64,
    num_heads=4,
    num_layers=2
)

# Generate!
prompt = "the cat sat on"
token_ids = vocab.encode(prompt)
generated = model.generate(token_ids, max_new_tokens=5)
print(vocab.decode(generated))
```

---

# 🎛️ Key Hyperparameters

| Parameter | Our Demo | GPT-3 | What it controls |
|-----------|----------|-------|------------------|
| embed_dim | 64 | 12,288 | Vector size |
| num_heads | 4 | 96 | Attention perspectives |
| num_layers | 2 | 96 | Depth of processing |
| vocab_size | ~30 | 50,257 | Word coverage |
| **Total params** | ~100K | 175B | Model capacity |

**Same architecture, different scale!**

---

# 🔥 Temperature Control

Control randomness in generation:

```python
probs = softmax(logits / temperature)
```

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.1 | Very focused, repetitive | Factual responses |
| 1.0 | Balanced | General use |
| 2.0 | Creative, chaotic | Brainstorming |

---

# 📱 Interactive Demo

Let's explore with Streamlit!

```bash
streamlit run app.py
```

**Try:**
1. Generate text with different prompts
2. Visualize attention patterns
3. Experiment with temperature
4. Compare layer outputs

---

# 🧪 Code Walkthrough: Positional Encoding

```python
class PositionalEncoding:
    def __init__(self, embed_dim, max_seq_len=512):
        # Create position indices
        position = np.arange(max_seq_len)[:, np.newaxis]
        
        # Create dimension indices with scaling
        div_term = np.exp(
            np.arange(0, embed_dim, 2) * 
            -(np.log(10000.0) / embed_dim)
        )
        
        # Compute sine and cosine
        self.encoding = np.zeros((max_seq_len, embed_dim))
        self.encoding[:, 0::2] = np.sin(position * div_term)
        self.encoding[:, 1::2] = np.cos(position * div_term)
```

---

# 🧪 Code Walkthrough: Multi-Head Attention

```python
def forward(self, x, causal=True):
    # Project to Q, K, V
    Q = x @ self.W_q
    K = x @ self.W_k  
    V = x @ self.W_v
    
    # Split into heads
    Q = Q.reshape(seq_len, num_heads, head_dim)
    
    # Compute attention for each head
    for h in range(num_heads):
        scores = Q[:, h] @ K[:, h].T / sqrt(head_dim)
        if causal:
            scores += causal_mask  # -inf for future
        weights[h] = softmax(scores)
        head_outputs[h] = weights[h] @ V[:, h]
    
    # Combine heads
    return concat(head_outputs) @ W_out
```

---

# 🧪 Code Walkthrough: Generation Loop

```python
def generate(self, token_ids, max_new_tokens=10):
    for _ in range(max_new_tokens):
        # Forward pass
        logits, _ = self.forward(token_ids)
        
        # Get last position's predictions
        next_logits = logits[-1]
        
        # Apply temperature and sample
        probs = softmax(next_logits / temperature)
        next_token = np.random.choice(len(probs), p=probs)
        
        # Append and continue
        token_ids.append(next_token)
    
    return token_ids
```

---

# 🎯 Key Takeaways

1. **Transformers = Attention + FFN + Residuals + LayerNorm**
   - Same building blocks, stacked N times

2. **Position matters** — Positional encoding tells the model word order

3. **Multi-head = Multiple perspectives** — Different heads learn different patterns

4. **Residual connections are critical** — Enable deep networks to train

5. **Scale is what makes LLMs powerful** — Same architecture, billions of parameters

---

# 🔗 Workshop Connections

```
Workshop 1 (Tokenization)
         ↓
    Words → Token IDs
         ↓
Workshop 2 (Embeddings)  
         ↓
    Token IDs → Vectors
         ↓
Workshop 4 (Attention)
         ↓
    Vectors → Context-aware Vectors
         ↓
Workshop 5 (Transformers) ← YOU ARE HERE
         ↓
    Full architecture: Embed → Attend → Transform → Predict
         ↓
Workshop 6 (RAG)
         ↓
    Add external knowledge retrieval
```

---

# ➡️ Next Workshop

## Workshop 6: RAG (Retrieval-Augmented Generation)

**The alien gets a search engine!**

- Connect our transformer to a vector database
- Retrieve relevant documents for context
- Build a simple Q&A system

**Coming up:** The complete GenAI pipeline!

---

# 🙋 Q&A Time

Common questions:
- Why not use RNNs instead?
- How does ChatGPT handle long conversations?
- What's the difference between GPT and BERT?
- How many GPUs does it take to train GPT-4?

---

# 📚 Resources

- **Original Paper:** "Attention Is All You Need" (2017)
- **Illustrated Transformer:** jalammar.github.io/illustrated-transformer/
- **Our Code:** `workshops/05-transformers/`
- **Next Week:** RAG — Giving LLMs external memory

---

# 🎉 Thank You!

## Workshop 5 Complete!

**Files to explore:**
- `transformer.py` — Full implementation
- `app.py` — Interactive demo
- `test_transformer.py` — 25 passing tests

**See you in Workshop 6: RAG!**

*GenAI Self-Build Series — Demystifying AI, one component at a time*
