# ❓ Workshop 1: Q&A Reference

> **Common questions and answers about tokenization**

---

## 🛸 The Big Picture: The Alien Analogy

Imagine an alien lands on Earth and needs to learn English. They don't know what words are—they just see a stream of symbols:

```
"Thequickbrownfoxjumpsoverthelazydog"
```

The alien notices patterns:
- `"the"` appears a lot → assign it one symbol: `θ`
- `"ing"` is common at word ends → assign it: `ω`
- `"tion"` shows up frequently → assign it: `τ`

After studying millions of texts, the alien builds a **codebook** of common patterns. Now they can read English much faster—not letter by letter, but in meaningful chunks!

**This is exactly what BPE tokenization does.** The "alien" is our algorithm, and the "codebook" is our vocabulary.

---

## Conceptual Questions

### Q1: Why can't we just use ASCII codes?

**Answer:** While ASCII does convert characters to numbers, it has several limitations:

1. **No semantic meaning**: ASCII code 72 ('H') doesn't tell the model anything about its meaning
2. **Fixed mapping**: We can't optimize the representation for our specific task
3. **No vocabulary control**: Every character is a separate token, including rare ones
4. **Inefficiency**: Common words like "the" become 3 tokens instead of 1

Tokenization with vocabulary learning allows the model to create more efficient, task-appropriate representations.

---

### Q2: How does GPT-4's tokenizer work?

**Answer:** GPT-4 uses a BPE (Byte Pair Encoding) tokenizer called `tiktoken` with approximately 100,000 tokens. Key features:

- **Byte-level BPE**: Works on bytes, not characters (handles any Unicode)
- **Trained on massive corpus**: Learned merges from internet-scale text
- **Special tokens**: Has tokens for padding, end-of-text, etc.

```python
# Try it yourself!
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
enc.encode("Hello, world!")  # [9906, 11, 1917, 0]
```

The token `9906` represents "Hello" because it's a very common word.

---

### Q3: Why do LLMs struggle with math and spelling?

**Answer:** This is directly related to tokenization! 

**🧩 Analogy: The Jigsaw Puzzle**

Imagine you're given a jigsaw puzzle, but some pieces have been glued together. You can see the whole picture, but you can't separate the individual pieces to count them.

That's what happens with tokenization:

**Math problems:**
```python
# "127 + 456" might tokenize as:
["127", " +", " 456"]  # or even ["12", "7", " +", " 4", "56"]
```
The model sees glued-together puzzle pieces, not individual digits. Arithmetic becomes like doing math while only seeing numbers as blurry groups!

**Spelling problems:**
```python
# "strawberry" might tokenize as:
["straw", "berry"]  # The model doesn't "see" individual letters!
```
When asked "How many r's in strawberry?", it's like asking someone to count the small pieces inside a glued chunk—they have to guess!

---

### Q4: What's the ideal vocabulary size?

**Answer:** It depends on the use case, but here are common ranges:

**📚 Analogy: The Library Card Catalog**

Think of vocabulary size like organizing a library:
- **Too few categories (small vocab)**: "Science" contains everything from biology to physics to chemistry. Finding a specific book takes forever (long sequences).
- **Too many categories (large vocab)**: Every book has its own category. The catalog is huge and hard to maintain (memory issues).
- **Just right (BPE)**: Categories like "Biology > Marine > Fish" give you quick access without overwhelming complexity.

| Use Case | Vocab Size | Reasoning |
|----------|-----------|-----------|
| Character-level | 100-300 | All characters |
| English only | 30,000-50,000 | Cover most words |
| Multilingual | 100,000-250,000 | Multiple languages |
| Specialized domain | 50,000-100,000 | Domain terms + general |

**Trade-offs:**
- Smaller vocab → Longer sequences, less memory
- Larger vocab → Shorter sequences, more memory, needs more training data

Modern LLMs typically use 32,000-100,000 tokens.

---

### Q5: How do multilingual models tokenize?

**Answer:** Multilingual models like mBERT or GPT-4 use:

1. **Shared BPE vocabulary**: Trained on text from many languages
2. **Language-agnostic subwords**: Common patterns across languages merge
3. **Larger vocabularies**: 100,000+ tokens to cover many languages

**Challenge:** Tokenization isn't equally efficient across languages:
```python
# English: "Hello" → 1 token
# Japanese: "こんにちは" → might be 3-5 tokens
```
This means non-English languages often require more tokens for the same content.

---

### Q6: What's the difference between BPE, WordPiece, and SentencePiece?

**Answer:**

| Algorithm | Used By | Key Difference |
|-----------|---------|----------------|
| **BPE** | GPT, LLaMA | Merges based on frequency |
| **WordPiece** | BERT | Merges based on likelihood increase |
| **SentencePiece** | T5, LLaMA | Treats text as raw characters (no pre-tokenization) |
| **Unigram** | Some models | Statistical model, prunes vocabulary |

For this workshop, we focus on BPE because it's the most common and intuitive.

---

## Practical Questions

### Q7: My tokenizer gives different IDs than expected. Why?

**Answer:** Token IDs depend on:

1. **Training corpus**: Different text → different vocabulary
2. **Vocabulary size**: Smaller vocab may not include rare words
3. **Order of insertion**: IDs are assigned based on when tokens are added

**Debugging tip:**
```python
print(tokenizer.vocab)  # See your actual vocabulary
print(tokenizer.encode("test"))  # Check what IDs are produced
```

---

### Q8: How do I handle special characters and emojis?

**Answer:** For our simple tokenizer:

**Character tokenizer:** Works automatically (every character gets an ID)

**Word tokenizer:** Our regex `\b\w+\b` skips special characters. To include them:
```python
# Option 1: Include in regex
words = re.findall(r'\S+', text.lower())  # Any non-whitespace

# Option 2: Add special handling
if '😀' in text:
    words.append('<EMOJI>')
```

**Production tokenizers:** Handle Unicode and emojis through byte-level encoding.

---

### Q9: Why does my BPE tokenizer seem slow?

**Answer:** BPE training is O(n × m) where n = corpus size, m = number of merges. Speed tips:

1. **Pre-compute word frequencies**: Don't re-tokenize each iteration
2. **Use efficient data structures**: `Counter` and dictionaries
3. **Limit vocabulary size**: Fewer merges = faster training

For large corpora, production implementations use:
- Rust/C++ implementations (like HuggingFace tokenizers)
- Parallelization
- Caching

---

### Q10: What happens if I encode text that has characters not in my vocabulary?

**Answer:** Depends on the strategy:

**Character tokenizer:**
```python
# Option 1: Skip unknown characters
encoded = [vocab[c] for c in text if c in vocab]

# Option 2: Use default ID
encoded = [vocab.get(c, vocab.get('<UNK>', 0)) for c in text]
```

**Word tokenizer:**
```python
# Always use <UNK> for unknown words
encoded = [vocab.get(word, 0) for word in words]  # 0 = <UNK>
```

**Best practice:** Always include an `<UNK>` token in your vocabulary!

---

### Q11: Can I save and load my trained tokenizer?

**Answer:** Yes! Here's how:

```python
import json

# Save
def save_tokenizer(tokenizer, filepath):
    data = {
        'strategy': tokenizer.strategy,
        'vocab': tokenizer.vocab,
        'merges': tokenizer.merges  # For BPE
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)

# Load
def load_tokenizer(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    tokenizer = SimpleTokenizer(strategy=data['strategy'])
    tokenizer.vocab = data['vocab']
    tokenizer.inverse_vocab = {int(v): k for k, v in data['vocab'].items()}
    tokenizer.merges = [tuple(m) for m in data.get('merges', [])]
    return tokenizer
```

---

### Q12: How do production tokenizers handle the `</w>` end marker?

**Answer:** Different approaches:

1. **Explicit marker** (what we use): `</w>` at end of each word
2. **Prefix marker**: `Ġ` (byte 32) before space-preceded tokens (GPT-2/3)
3. **Special byte**: Byte-level representation handles spaces naturally

Example from GPT-2:
```python
# "Hello world" tokenizes as:
["Hello", "Ġworld"]  # Ġ indicates "preceded by space"
```

---

## Deep Dive Questions

### Q13: Why does BPE scale by √d in attention (for future workshops)?

**Answer:** This is actually about attention mechanisms (Workshop 4), not tokenization! But briefly:

The scaling factor `1/√d_k` prevents dot products from becoming too large when dimensions are high, which would make softmax produce very peaked distributions.

---

### Q14: How does tokenization affect model performance?

**Answer:** Significantly! Research shows:

1. **Fertility** (tokens per word): Lower is generally better
2. **Coverage**: Higher vocabulary coverage = fewer <UNK> tokens
3. **Consistency**: Same words should tokenize the same way
4. **Language efficiency**: Some tokenizers work better for certain languages

Poor tokenization can hurt model performance even with perfect architecture.

---

### Q15: What's next after tokenization?

**Answer:** After tokens, we need **embeddings** (Workshop 2)!

**🍳 Analogy: Recipe Ingredients → Flavor Profiles**

Tokenization is like converting a recipe into a list of ingredient IDs:
```
"chocolate cake" → [ingredient_23, ingredient_89]
```

But IDs don't tell you anything about *taste*! Embeddings are like converting those IDs into flavor profiles:
```
ingredient_23 (chocolate) → [sweet: 0.8, bitter: 0.3, rich: 0.9, ...]
ingredient_89 (cake)      → [sweet: 0.6, fluffy: 0.8, baked: 0.7, ...]
```

Now the model understands that chocolate and cocoa are similar (both bitter+sweet), even though they have different IDs!

```
Text → Tokens → Embeddings → Neural Network → Output
```

Token IDs are just integers with no meaning. Embeddings convert these IDs into dense vectors that capture semantic meaning:

```python
token_id = 42           # Integer, no meaning
embedding = [0.2, -0.5, 0.1, ...]  # 50-dimensional vector with meaning!
```

That's exactly what we'll build in Workshop 2!

---

## 🎯 Quick Answers

| Question | Short Answer |
|----------|--------------|
| Best tokenizer type? | BPE for most cases |
| Vocab size for English? | 30,000-50,000 |
| Handle unknown words? | Use `<UNK>` token (ID 0) |
| Why BPE over word-level? | Handles OOV better |
| Case sensitive? | Usually lowercase for simplicity |
| Include punctuation? | Depends on use case |

---

*Have more questions? Ask during the Q&A session or reach out afterward!*

*Workshop 1 of 6 | GenAI Self-Build Series*
