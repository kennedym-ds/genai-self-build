# 🔍 Under the Hood: How GenAI Self-Build Really Works

## Purpose of This Guide

This document reveals the **internals** of each workshop's implementation—what's happening behind the scenes, what's been simplified for education, and how it compares to production systems.

**Use this when you want to:**
- Understand the actual algorithms being used
- See step-by-step execution traces
- Compare educational vs. production implementations
- Debug or extend the code
- Prepare for technical interviews

---

## Table of Contents

1. [Debug Mode: Seeing Inside the Black Box](#debug-mode)
2. [Workshop 1: Tokenization Internals](#workshop-1-tokenization)
3. [Workshop 2: Embeddings Internals](#workshop-2-embeddings)
4. [Workshop 3: Vector Database Internals](#workshop-3-vector-database)
5. [Workshop 4: Attention Mechanism Internals](#workshop-4-attention)
6. [Workshop 5: Transformer Internals](#workshop-5-transformer)
7. [Workshop 6: RAG Pipeline Internals](#workshop-6-rag)
8. [What's Simplified and Why](#simplifications)
9. [Production System Comparison](#production-comparison)

---

## Debug Mode: Seeing Inside the Black Box {#debug-mode}

All implementations support a **debug mode** that shows step-by-step execution:

```python
from tokenizer import SimpleTokenizer

# Enable debug mode to see internals
tokenizer = SimpleTokenizer(strategy='word', debug=True)
tokenizer.train(corpus)  # Prints detailed training steps
tokens = tokenizer.encode("Hello world")  # Shows encoding process
```

### What Debug Mode Shows

- **Input processing**: What data is being fed in
- **Intermediate steps**: Algorithm state at each iteration
- **Data transformations**: How inputs become outputs
- **Statistics**: Counts, sizes, performance metrics
- **Decisions**: Why certain paths are chosen

Example output:
```
============================================================
🔍 WORD TOKENIZER TRAINING - UNDER THE HOOD
============================================================
📥 Input: 6 documents
🎯 Target vocabulary size: 100

📊 Step 1: Tokenized all documents
   Total words: 234
   First 10 words: ['the', 'quick', 'brown', 'fox', ...]

🎯 Step 2: Counted word frequencies
   Unique words: 87
   Top 10 most common:
      'the': 23 times
      'is': 15 times
      'to': 12 times
      ...

✅ Step 5-6: Built final vocabulary
   Final vocabulary size: 89
   Coverage: 87/87 unique words (100.0% of unique words)
============================================================
```

---

## Workshop 1: Tokenization Internals {#workshop-1-tokenization}

### Algorithm Complexity

| Strategy | Training | Encoding | Space |
|----------|----------|----------|-------|
| Character | O(n) | O(n) | O(k) where k = unique chars (~100) |
| Word | O(n) | O(n) | O(V) where V = vocab size (10k-100k) |
| BPE | O(n² × m) | O(n × m) | O(V) where m = num merges |

### Character Tokenizer: Step-by-Step

**Input:** `"Hello world!"`

**Step 1: Find unique characters**
```python
text = "Hello world!"
unique_chars = {'H', 'e', 'l', 'o', ' ', 'w', 'r', 'd', '!'}
# Sort for reproducibility: [' ', '!', 'H', 'd', 'e', 'l', 'o', 'r', 'w']
```

**Step 2: Assign IDs**
```python
vocab = {
    ' ': 0, '!': 1, 'H': 2, 'd': 3, 'e': 4,
    'l': 5, 'o': 6, 'r': 7, 'w': 8
}
```

**Step 3: Encode**
```python
"Hello world!" → [2, 4, 5, 5, 6, 0, 8, 6, 7, 5, 3, 1]
```

**Pros:**
- Simple, always works
- Small vocabulary (~100 tokens)
- No unknown tokens

**Cons:**
- Long sequences (1 char = 1 token)
- Loses word structure
- Inefficient for neural networks

---

### Word Tokenizer: Step-by-Step

**Input:** `["The cat sat.", "The dog ran."]`

**Step 1: Tokenize into words**
```python
# Using regex: \b\w+\b (word boundaries)
doc1 → ["the", "cat", "sat"]
doc2 → ["the", "dog", "ran"]
all_words = ["the", "cat", "sat", "the", "dog", "ran"]
```

**Step 2: Count frequencies**
```python
Counter({'the': 2, 'cat': 1, 'sat': 1, 'dog': 1, 'ran': 1})
```

**Step 3: Build vocabulary (most common first)**
```python
vocab = {
    '<UNK>': 0,  # Unknown words
    '<PAD>': 1,  # Padding
    'the': 2,    # Most frequent
    'cat': 3,
    'sat': 4,
    'dog': 5,
    'ran': 6
}
```

**Step 4: Encode**
```python
"The cat sat" → [2, 3, 4]
"The dog barked" → [2, 5, 0]  # "barked" → <UNK> (not in vocab)
```

**Pros:**
- Intuitive (words = tokens)
- Short sequences
- Good for languages with clear word boundaries

**Cons:**
- Huge vocabulary (100k+ words)
- Out-of-vocabulary (OOV) problem
- Can't handle misspellings

---

### BPE Tokenizer: Step-by-Step

**The Key Insight:** Merge frequently co-occurring character pairs iteratively.

**Input:** `["hello", "hello", "world"]`

**Iteration 0: Start with characters**
```python
word_freqs = {
    ('h','e','l','l','o','</w>'): 2,  # "hello" appears twice
    ('w','o','r','l','d','</w>'): 1,
}
```

**Iteration 1: Find most frequent pair**
```python
pair_freqs = {
    ('h','e'): 2,
    ('e','l'): 2,
    ('l','l'): 2,  # ← Most frequent
    ('l','o'): 2,
    ('w','o'): 1,
    ...
}
# Merge ('l','l') → 'll'
word_freqs = {
    ('h','e','ll','o','</w>'): 2,
    ('w','o','r','l','d','</w>'): 1,
}
vocab.add('ll')
```

**Iteration 2: Continue merging**
```python
# Next most frequent might be ('h','e') → 'he'
word_freqs = {
    ('he','ll','o','</w>'): 2,
    ('w','o','r','l','d','</w>'): 1,
}
vocab.add('he')
```

**Final vocabulary (example):**
```python
vocab = ['h', 'e', 'l', 'o', 'w', 'r', 'd', '</w>',  # characters
         'll', 'he', 'llo', 'hello', ...]              # learned merges
```

**Encoding "hello world":**
```python
# Apply learned merges in order
"hello" → "he" + "llo" → "hello" (if fully merged)
"world" → "w" + "o" + "r" + "l" + "d"
Result: ["hello", "w", "o", "r", "l", "d"]
```

**Pros:**
- Balances vocab size and sequence length
- Handles rare/unknown words (falls back to chars)
- Used by GPT, BERT, LLaMA

**Cons:**
- More complex algorithm
- Training is slower (O(n²))
- Requires tuning vocab size

---

## Statistics and Metrics

### Using `get_stats()`

```python
tokenizer = SimpleTokenizer(strategy='word')
tokenizer.train(corpus, vocab_size=1000)

stats = tokenizer.get_stats()
print(stats)
```

**Output:**
```python
{
    'strategy': 'word',
    'vocab_size': 842,
    'total_chars': 15234,
    'total_words': 2456,
    'unique_words': 842,
    'compression_ratio': 6.2  # chars per word
}
```

### Key Metrics Explained

- **vocab_size**: How many unique tokens in vocabulary
- **compression_ratio**: How much text is compressed
  - Character: ~1.0 (no compression)
  - Word: ~5-7 (words are ~5-7 chars)
  - BPE: ~3-5 (subwords)
- **coverage**: % of unique words in vocabulary
  - 100% = no OOV tokens
  - <100% = rare words become `<UNK>`

---

## What's Simplified (vs Production) {#simplifications}

### What We Kept Real
✅ Core algorithms are authentic (BPE, word frequency, etc.)
✅ Same tradeoffs (vocab size vs sequence length)
✅ Same data structures (vocab dicts, inverse lookups)
✅ Same encoding/decoding logic

### What We Simplified
⚠️ **No byte-level encoding** — Real BPE works on bytes, not chars (handles all Unicode)
⚠️ **No special token handling** — Real tokenizers have `<EOS>`, `<BOS>`, `<MASK>`, etc.
⚠️ **No regex pre-tokenization** — GPT uses complex regex before BPE
⚠️ **No optimization** — Real tokenizers use Rust/C++ for speed
⚠️ **No vocabulary pruning** — Real systems prune rare tokens
⚠️ **Smaller vocab sizes** — We use 100-1000, real uses 50k-100k

---

## Production System Comparison {#production-comparison}

### GPT-4 Tokenizer (tiktoken)

| Feature | Our BPE | GPT-4 (tiktoken) |
|---------|---------|------------------|
| Vocab Size | 100-1000 | 100,000 |
| Algorithm | Character-based BPE | Byte-level BPE |
| Speed | Python (slow) | Rust (10000x faster) |
| Special tokens | Basic (<UNK>, <PAD>) | Many (<|endoftext|>, <|im_start|>, etc.) |
| Pre-tokenization | Simple word split | Complex regex patterns |

**How to use the real thing:**
```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
tokens = encoder.encode("Hello world!")
print(tokens)  # [9906, 1917, 0]
```

### Comparing Output

**Our BPE tokenizer:**
```python
"The quick brown fox" → [2, 54, 12, 89, 45]  # 5 tokens
```

**GPT-4 tokenizer:**
```python
"The quick brown fox" → [791, 4062, 14198, 39935]  # 4 tokens
# More efficient due to larger vocab and byte-level encoding
```

---

## Debugging Tips

### Enable Debug Mode Everywhere

```python
# See what's happening in training
tokenizer = SimpleTokenizer(strategy='bpe', debug=True)
tokenizer.train(large_corpus, vocab_size=500)

# Check statistics
stats = tokenizer.get_stats()
print(f"Vocabulary covers {stats['unique_words']} unique words")
print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
```

### Visualize Vocabulary

```python
# Print all vocab mappings
for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1])[:50]:
    print(f"{idx:4d}: '{token}'")
```

### Test Roundtrip

```python
original = "Test this text!"
encoded = tokenizer.encode(original)
decoded = tokenizer.decode(encoded)
assert decoded == original.lower(), f"Roundtrip failed: {decoded}"
```

---

## Further Reading

**Production Tokenizers:**
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) - Fast, multilingual
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's BPE tokenizer
- [SentencePiece](https://github.com/google/sentencepiece) - Google's tokenizer

**Papers:**
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2015) - Original BPE paper
- "Language Models are Unsupervised Multitask Learners" (GPT-2 paper) - Byte-level BPE

**Interactive Tools:**
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) - Visualize GPT tokenization
- [Hugging Face Tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) - Compare different tokenizers

---

## Next: Workshop 2 - Embeddings

Continue to [Embeddings Under the Hood](./UNDER_THE_HOOD_EMBEDDINGS.md) to see how tokens become meaningful vectors in high-dimensional space.
