---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #ffffff
---

<!-- _class: lead -->

# 📚 Workshop 3: Vector Databases
## *The alien's library with magic shelves*

### GenAI Self-Build Series (3 of 6)

---

# 🎯 Learning Objectives

By the end of this workshop, you will understand:

1. **Why** we need specialized databases for vectors
2. **Three indexing strategies**: Flat, LSH, and IVF
3. **Trade-offs** between speed, accuracy, and memory
4. **How** production vector databases work

---

# 📖 Previously on GenAI Self-Build...

**Workshop 1: Tokenization** 🛸
- Alien learned to read symbols → tokens
- Text becomes numbers

**Workshop 2: Embeddings** 🗺️
- Alien created a meaning map
- Similar words live close together

**Today's Question:**
> With MILLIONS of word vectors, how do we find similar ones FAST?

---

# 🤔 The Problem

Imagine searching for books similar to "Introduction to AI":

| Library Size | Books to Check | Time |
|--------------|----------------|------|
| 100 books | 100 | Instant |
| 10,000 books | 10,000 | A few seconds |
| 1 million books | 1,000,000 | Minutes! |
| 1 billion books | 1,000,000,000 | Hours!! |

**Linear search doesn't scale!**

---

# 📚 The Library Analogy

The alien's library has 1 MILLION books (vectors).

**Brute Force Approach:**
```
Walk to shelf 1, check book... no match
Walk to shelf 2, check book... no match
Walk to shelf 3, check book... no match
... (999,997 more times)
```

**Smart Librarian Approach:**
```
"You want AI books? That's Section C!"
Walk directly to Section C
Check only 1,000 books in that section
```

**1000x faster!**

---

# 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    📚 Vector Database                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   📄 Vectors   →   📊 Index   →   🔍 Search   →   🎯     │
│   (stored)         (organized)     (query)        Results │
│                                                          │
├──────────────────────────────────────────────────────────┤
│              Index Strategy Options:                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│   │📖 Flat  │    │🎲 LSH   │    │📊 IVF   │             │
│   │ Exact   │    │ Hashing │    │Clusters │             │
│   └─────────┘    └─────────┘    └─────────┘             │
└──────────────────────────────────────────────────────────┘
```

---

# 📖 Strategy 1: Flat (Brute Force)

### The Simplest Approach

```python
def search(query, vectors):
    scores = []
    for vector in vectors:  # Check EVERY vector
        score = cosine_similarity(query, vector)
        scores.append(score)
    return top_k(scores)
```

| Pros | Cons |
|------|------|
| ✅ 100% accurate | ❌ O(n) - slow |
| ✅ Simple | ❌ Doesn't scale |
| ✅ No index needed | ❌ Memory intensive |

**Use when:** < 10,000 vectors

---

# 🎲 Strategy 2: LSH (Locality Sensitive Hashing)

### The Magic Coin Analogy

Imagine a magic coin that:
- Lands **HEADS** for "happy" 🪙 ➡️ H
- Lands **HEADS** for "joyful" 🪙 ➡️ H
- Lands **TAILS** for "sad" 🪙 ➡️ T

**Similar items get similar coin flips!**

With multiple coins: `happy → HHTH`, `joyful → HHTH`, `sad → TTHT`

Now just look in the "HHTH" bucket!

---

# 🎲 LSH: How It Works

```
Step 1: Create random "coins" (hyperplanes)
        ─────────────────────────
        
Step 2: For each vector, flip all coins
        "happy"  → [1,0,1,1] → bucket "1011"
        "joyful" → [1,0,1,1] → bucket "1011"  (same!)
        "sad"    → [0,1,0,0] → bucket "0100"
        
Step 3: At query time
        Query "glad" → [1,0,1,1] → bucket "1011"
        Only search bucket "1011"!
```

---

# 🎲 LSH: Performance

| Pros | Cons |
|------|------|
| ✅ Very fast lookup | ❌ Approximate |
| ✅ Works for any dimension | ❌ Tuning required |
| ✅ Memory efficient | ❌ Hash collisions |

**Parameters:**
- `num_tables`: More tables = better recall, slower
- `num_bits`: More bits = smaller buckets, more precise

---

# 📊 Strategy 3: IVF (Inverted File Index)

### The Section System

```
Library Sections:
┌─────────────────────────────────────────────────┐
│  📚 Section A      📚 Section B      📚 Section C │
│  (Romance)        (Sci-Fi)          (Technical)  │
│                                                  │
│  ● ● ●            ● ● ●             ● ● ● ●      │
│  ● ● ●            ● ● ●             ● ● ● ●      │
└─────────────────────────────────────────────────┘

Query: "Machine Learning textbook"
→ Go to Section C (Technical)
→ Search only those books
```

---

# 📊 IVF: How It Works

```
Step 1: TRAINING (k-means clustering)
        Divide 1M vectors into 100 clusters
        Each cluster has ~10,000 vectors
        
Step 2: INDEXING
        Assign each vector to its nearest cluster
        Store in inverted file structure
        
Step 3: SEARCH
        Find nearest cluster(s) to query
        Search only those clusters
        nprobe=1: Search 1 cluster (fast, ~90% recall)
        nprobe=5: Search 5 clusters (slower, ~99% recall)
```

---

# 📊 IVF: Performance

| Pros | Cons |
|------|------|
| ✅ Good speed/accuracy | ❌ Training needed |
| ✅ Tunable via nprobe | ❌ Approximate |
| ✅ Industry standard | ❌ Cluster updates tricky |

**Parameters:**
- `num_clusters`: More = faster search, more training
- `nprobe`: How many clusters to check (speed/accuracy)

---

# 🆚 Strategy Comparison

|  | Flat | LSH | IVF |
|--|------|-----|-----|
| **Build** | O(1) | O(n × tables) | O(n × k-means) |
| **Search** | O(n) | O(1) ~ O(n) | O(k × cluster) |
| **Accuracy** | 100% | 80-95% | 90-99% |
| **Memory** | Low | Medium | Medium |
| **Best for** | < 10K | High-dim | 100K-10M |

---

# 🎯 Live Demo Time!

Let's explore our vector database:

```bash
cd workshops/03-vector-databases
streamlit run app.py
```

We'll see:
1. 📊 How indexing affects search
2. 🏃 Speed vs accuracy trade-offs
3. 🎨 Vector space visualization

---

# 🌍 Real-World Vector Databases

| Database | Primary Algorithm | Notable Users |
|----------|------------------|---------------|
| **Pinecone** | IVF + HNSW | OpenAI, Notion |
| **Weaviate** | HNSW | Various startups |
| **Milvus** | Multiple options | Enterprise AI |
| **FAISS** | All of above | Meta, Research |
| **Chroma** | HNSW | LangChain apps |

Most use **HNSW** (graph-based) - even faster than IVF!

---

# 🔗 Connection to RAG (Workshop 6)

```
┌──────────────────────────────────────────────────────┐
│                   RAG Pipeline                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  User Question                                       │
│       │                                              │
│       ▼                                              │
│  [Embedding Model] ──────────────┐                   │
│       │                          │                   │
│       ▼                          ▼                   │
│  Query Vector  ──→  📚 Vector DB  ──→  Context      │
│                         ▲                            │
│                         │                            │
│                   Documents                          │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**The Vector DB is the retrieval engine for RAG!**

---

# 💡 Key Insights

1. **Linear search doesn't scale**
   - 1M vectors × 1M queries = 1 trillion operations

2. **Approximate is usually OK**
   - 95% recall is fine for most applications
   - Users won't notice the 5% difference

3. **It's all about trade-offs**
   - More accuracy = more time
   - More speed = less accuracy
   - Choose based on your use case

---

# 🎯 Key Takeaways

1. **Vector databases** solve the "needle in a haystack" problem for AI

2. **Three main strategies:**
   - 📖 Flat: Exact but slow
   - 🎲 LSH: Hash similar items together
   - 📊 IVF: Organize into searchable clusters

3. **Production systems** use hybrid approaches (HNSW + IVF)

---

# ➡️ Next Workshop: Attention 👀

Now we know:
- ✅ How to tokenize text
- ✅ How to embed tokens into vectors
- ✅ How to find similar vectors fast

**Next question:**
> How does the model know what to focus on?

In "The cat sat on the mat", why does "sat" relate more to "cat" than "mat"?

**That's the attention mechanism!**

---

<!-- _class: lead -->

# 🙋 Q&A Time!

## Questions?

---

# 📚 Resources

- **Code**: `workshops/03-vector-databases/`
- **Cheatsheet**: `cheatsheet.md`
- **Q&A**: `qna.md`
- **FAISS Tutorial**: [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)

### Workshop 3 of 6 | GenAI Self-Build Series
*The alien's library with magic shelves - finding similar things fast!* 📚
