# 🔍 Workshop 6: RAG — The Grand Finale

## 🛸 The Search Engine Analogy

Our alien's brain (the transformer) has a problem: it only knows what it learned in "school" (training data). Ask about something new, and it might confidently make things up!

**Solution:** Give the alien a library card! 📚🔍

Instead of relying on memory, the alien can now:
1. 📝 Receive a question
2. 🔍 Search the library for relevant books
3. 📚 Pull out the most relevant pages
4. 🧠 Read pages + question together
5. 💬 Answer based on what was just read

This is **RAG** - Retrieval-Augmented Generation. It's like giving someone an **open-book exam** instead of a **closed-book exam**!

## 📋 What You'll Learn

- How to embed and store documents for retrieval
- How vector search finds semantically similar content
- How to augment prompts with retrieved context
- How the complete RAG pipeline works end-to-end
- Why RAG solves hallucination, staleness, and opacity

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Question   │ ──▶ │   Embed     │ ──▶ │   Search    │
│ "What is X?"│     │ [0.2, 0.8,] │     │  Vector DB  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │ ◀── │  Generate   │ ◀── │  Augment    │
│ "X is..." + │     │    (LLM)    │     │   Prompt    │
│   Sources   │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
streamlit run app.py

# Run tests
python test_rag.py
```

## 📁 Files

| File | Description |
|------|-------------|
| `rag.py` | Complete RAG implementation with all components |
| `app.py` | Interactive Streamlit demo |
| `test_rag.py` | Comprehensive test suite (25 tests) |
| `cheatsheet.md` | Quick reference guide |
| `qna.md` | Common questions answered |
| `slides/slides.md` | Marp presentation |

## 🐍 Core Components

### SimpleEmbedder
Converts text to vectors using bag-of-words with random projection.

```python
embedder = SimpleEmbedder(embed_dim=64)
vector = embedder.embed("What is Python?")
```

### DocumentStore
Vector database that stores and searches documents.

```python
store = DocumentStore(embedder)
store.add_documents(
    texts=["Python is a language...", "Java is..."],
    sources=["python.md", "java.md"]
)
results = store.search("programming", top_k=3)
```

### PromptBuilder
Creates augmented prompts with retrieved context.

```python
builder = PromptBuilder()
prompt = builder.build_prompt(question, retrieved_docs)
```

### RAGPipeline
Combines all components into a complete system.

```python
rag = RAGPipeline(embed_dim=64, top_k=3)
rag.add_knowledge(texts, sources)
result = rag.query("What is Python?")
print(result['answer'])
print(result['sources'])
```

## 🔗 Connection to LLMs

This workshop brings together **everything from the series**:

| Workshop | What We Built | Role in RAG |
|----------|---------------|-------------|
| 1. Tokenization | Text ↔ tokens | Process input/output |
| 2. Embeddings | Words → vectors | Encode meaning |
| 3. Vector DB | Store & search | Find similar docs |
| 4. Attention | Focus mechanism | Inside the transformer |
| 5. Transformer | Full architecture | Generate answers |
| **6. RAG** | **Complete pipeline** | **Combine everything!** |

**Real-world RAG systems:**
- **Perplexity AI** - Search engine with LLM answers
- **ChatGPT** - Web browsing, file uploads, custom GPTs
- **GitHub Copilot** - Code context retrieval
- **Enterprise AI** - Query internal knowledge bases

## ⚠️ Limitations of This Demo

Our simplified implementation:
- Uses bag-of-words embeddings (not neural embeddings)
- Uses extractive generation (not an actual LLM)
- Demonstrates concepts rather than production performance

Real systems use:
- Dense embeddings (OpenAI ada, Cohere, BGE)
- Actual LLMs for generation (GPT-4, Claude)
- Optimized vector stores (Pinecone, Weaviate, Qdrant)

## 🎓 Take-Home Exercises

1. **Hybrid Search**: Combine keyword matching with semantic search
2. **Reranking**: Add a second-pass reranker for better results
3. **Chunk Strategies**: Experiment with different document chunking
4. **Real Embeddings**: Swap in OpenAI or sentence-transformers embeddings

## 🎉 Congratulations!

You've completed the **GenAI Self-Build Workshop Series**!

You now understand the core components that power modern AI:
- ✅ How text becomes numbers (tokenization)
- ✅ How meaning becomes vectors (embeddings)
- ✅ How we find similar things (vector search)
- ✅ How models focus attention
- ✅ How transformers process language
- ✅ How RAG grounds generation in knowledge

**You've demystified GenAI! 🛸**
