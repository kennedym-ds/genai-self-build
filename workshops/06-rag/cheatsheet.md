# 🔍 RAG Cheatsheet

## Quick Definitions

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation - combining search with generation |
| **Retrieval** | Finding relevant documents from a knowledge base |
| **Augmentation** | Adding retrieved context to the prompt |
| **Generation** | Producing answers based on augmented prompt |
| **Knowledge Base** | Collection of documents to search |
| **Embedding** | Vector representation of text for similarity search |
| **Top-K** | Number of documents to retrieve |

---

## The RAG Pipeline

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Question  │ ──▶ │   Embed    │ ──▶ │   Search   │
│   "What    │     │ [0.2, 0.8, │     │  Top-K     │
│  is X?"    │     │  0.1, ...] │     │  Docs      │
└────────────┘     └────────────┘     └────────────┘
                                            │
                                            ▼
┌────────────┐     ┌────────────┐     ┌────────────┐
│   Answer   │ ◀── │  Generate  │ ◀── │  Augment   │
│  "X is..." │     │   (LLM)    │     │  Prompt    │
└────────────┘     └────────────┘     └────────────┘
```

---

## Core Code Snippets

### Create Embedder
```python
from rag import SimpleEmbedder
embedder = SimpleEmbedder(embed_dim=64)
vector = embedder.embed("Hello world")
```

### Create Document Store
```python
from rag import DocumentStore
store = DocumentStore(embedder)
store.add_documents(
    texts=["Doc 1", "Doc 2"],
    sources=["source1", "source2"]
)
results = store.search("query", top_k=3)
```

### Build Prompt
```python
from rag import PromptBuilder
builder = PromptBuilder(max_context_length=2000)
prompt = builder.build_prompt(question, retrieved_docs)
```

### Complete Pipeline
```python
from rag import RAGPipeline
rag = RAGPipeline(embed_dim=64, top_k=3)
rag.add_knowledge(texts, sources)
result = rag.query("What is Python?")
print(result['answer'])
print(result['sources'])
```

---

## Why RAG?

| Problem | RAG Solution |
|---------|--------------|
| **Hallucination** | Ground answers in retrieved docs |
| **Stale Knowledge** | Update knowledge base anytime |
| **No Sources** | Cite sources with every answer |
| **Context Limits** | Retrieve only relevant content |
| **Privacy** | Keep data in your own database |

---

## Key Metrics

- **Embedding Dimension**: Size of vector (64-1536 typical)
- **Top-K**: Number of docs to retrieve (3-10 typical)
- **Context Window**: Max chars in prompt (2000-128000)
- **Similarity Score**: Cosine similarity (0.0-1.0)

---

## 💡 Remember

> 🔍 **RAG = Search + Generate**
>
> Instead of asking an LLM to remember everything,
> we give it a search engine to find information first!
>
> This is like an **open-book exam** vs a **closed-book exam**.
> The model doesn't need to memorize - it just needs to know where to look.
