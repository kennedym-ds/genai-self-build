# 📚 Vector Database Cheatsheet

## 🛸 The Core Analogy
> The alien has a MILLION word vectors. Checking each one is like finding a book by checking EVERY shelf. Instead, build **magic shelves** that group similar items together!

---

## 🔧 Quick Reference

### Creating a Database

```python
from vector_db import SimpleVectorDB
import numpy as np

# Flat (brute force) - for small datasets
db = SimpleVectorDB(strategy='flat', dimensions=100)

# LSH - for very high dimensions
db = SimpleVectorDB(strategy='lsh', dimensions=100, 
                    num_tables=10, num_bits=8)

# IVF - for large datasets (most common)
db = SimpleVectorDB(strategy='ivf', dimensions=100,
                    num_clusters=20, nprobe=3)
```

### Core Operations

```python
# Add vectors
db.add("doc_1", vector1)                    # Single
db.add_batch(ids_list, vectors_array)       # Batch

# Build index (required for LSH/IVF)
db.build_index()

# Search
results = db.search(query_vector, top_k=10)
# Returns: [("doc_id", score), ...]

# Get info
db.size()                # Number of vectors
db.get_stats()           # Index statistics
```

---

## 📊 Strategy Decision Tree

```
                    How many vectors?
                          │
            ┌─────────────┼─────────────┐
            │             │             │
        < 10K         10K - 1M        > 1M
            │             │             │
            ▼             ▼             ▼
        📖 FLAT       📊 IVF      📊 IVF/HNSW
        (exact)     (clustered)   (production)
```

---

## 🆚 Quick Comparison

| Strategy | Time | Accuracy | Use Case |
|----------|------|----------|----------|
| 📖 Flat | O(n) | 100% | < 10K vectors |
| 🎲 LSH | O(1)~O(n) | ~80-95% | Very high dimensions |
| 📊 IVF | O(k × c) | ~90-99% | 100K - 10M vectors |

---

## 🎲 LSH Key Concepts

```
Random Hyperplanes → Binary Hash → Bucket Assignment

Vector [0.3, 0.8, -0.2] → Hash "1011" → Bucket #11

More tables = Better recall, Slower search
More bits = Smaller buckets, More precise matching
```

**Typical settings:** `num_tables=10, num_bits=8`

---

## 📊 IVF Key Concepts

```
Training: K-means creates cluster centroids
Indexing: Each vector assigned to nearest centroid
Search: Find nearest centroids, search those clusters

nprobe = 1  → Fast, ~90% recall
nprobe = 5  → Slower, ~99% recall
```

**Typical settings:** `num_clusters=sqrt(n), nprobe=5-20`

---

## 💡 Tips & Gotchas

✅ **Do:**
- Normalize vectors for cosine similarity
- Call `build_index()` before searching with LSH/IVF
- Use batch operations for large datasets
- Tune `nprobe` based on accuracy needs

❌ **Don't:**
- Use Flat for > 100K vectors
- Forget to rebuild index after adding vectors
- Expect 100% recall from approximate methods
- Use LSH with < 50 dimensions

---

## 🌍 Production Databases

| Database | Best For |
|----------|----------|
| **FAISS** | Research, flexibility |
| **Pinecone** | Managed, easy setup |
| **Weaviate** | Hybrid search |
| **Milvus** | Enterprise scale |
| **Chroma** | Local/LangChain |

---

## 🎯 Remember

> **Approximate ≠ Wrong**
> 
> Finding 95% of the best results in 1ms is better than
> finding 100% in 10 seconds for most AI applications!

---

*Workshop 3 of 6 | GenAI Self-Build Series*
*📚 The alien's library with magic shelves*
