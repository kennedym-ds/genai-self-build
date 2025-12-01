# GenAI Self-Build Workshop Series
## Demystifying AI Through Hands-On Construction

---

## Series Overview

**Mission:** Demystify Generative AI by having participants build core components from scratch, running everything locally with lightweight, educational implementations.

**Format:** 6 workshops × 1 hour each (45 min teaching/demo + 15 min Q&A)

**Target Audience:** Developers and technical learners who want to understand *how* GenAI works, not just how to use it.

**Prerequisites:**
- Basic Python proficiency (functions, loops, lists, dictionaries)
- Familiarity with command line
- No ML/AI experience required

---

## Learning Path

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Workshop 1  │───▶│ Workshop 2  │───▶│ Workshop 3  │
│ Tokenization│    │ Embeddings  │    │Vector Store │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
│ Workshop 6  │◀───│ Workshop 5  │◀───│ Workshop 4  │
│    RAG      │    │ Transformers│    │  Attention  │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## Workshop 1: Tokenization — Text to Numbers

### Learning Objectives
- Understand why computers need text converted to numbers
- Implement character-level, word-level, and BPE tokenization
- Compare trade-offs between tokenization strategies

### Session Breakdown

| Time | Activity |
|------|----------|
| 0-5 min | Hook: "How does ChatGPT read your message?" |
| 5-20 min | Theory: Tokenization strategies explained |
| 20-45 min | Build: Three tokenizers from scratch |
| 45-60 min | Q&A and wrap-up |

### Theory Concepts (~15 min)
- Why tokenization matters (vocabulary size, OOV handling)
- Character-level: Simple but long sequences
- Word-level: Intuitive but huge vocabularies
- Subword (BPE): The sweet spot used by GPT/BERT

### Hands-On Build (~25-30 min)

**Build Progression:**
1. **Character Tokenizer** (5 min)
   - Map each character to an integer
   - Encode/decode functions
   
2. **Word Tokenizer** (10 min)
   - Split on whitespace/punctuation
   - Build vocabulary from corpus
   - Handle unknown words

3. **Simple BPE Tokenizer** (15 min)
   - Start with character vocabulary
   - Iteratively merge most frequent pairs
   - Watch vocabulary evolve

### What You'll Build
```python
class SimpleTokenizer:
    def __init__(self, strategy='bpe'):
        self.vocab = {}
        self.merges = []
    
    def train(self, corpus, vocab_size=1000):
        # Build vocabulary from text
        pass
    
    def encode(self, text) -> list[int]:
        # Convert text to token IDs
        pass
    
    def decode(self, tokens) -> str:
        # Convert token IDs back to text
        pass
```

### Key Takeaways
- ✅ All LLMs start by converting text to numbers
- ✅ BPE balances vocabulary size with sequence length
- ✅ Tokenization affects model behavior (why some models struggle with math)

### Tech Stack
- Python 3.10+
- No external libraries (pure Python)
- Optional: `collections.Counter` for efficiency

### Deliverables
- [ ] Starter code template with TODOs
- [ ] Reference implementation
- [ ] Slide deck (10-12 slides)
- [ ] Take-home: Tokenize different languages, observe differences

---

## Workshop 2: Embeddings — Meaning in Math

### Learning Objectives
- Understand how meaning is captured in vectors
- Build word embeddings using co-occurrence
- Visualize semantic relationships in vector space

### Session Breakdown

| Time | Activity |
|------|----------|
| 0-5 min | Hook: "King - Man + Woman = Queen" demo |
| 5-20 min | Theory: Distributional semantics |
| 20-45 min | Build: Co-occurrence embeddings |
| 45-60 min | Q&A and wrap-up |

### Theory Concepts (~15 min)
- Distributional hypothesis: "Words are known by the company they keep"
- One-hot encoding vs. dense vectors
- How Word2Vec/GloVe capture meaning
- Cosine similarity for measuring relatedness

### Hands-On Build (~25-30 min)

**Build Progression:**
1. **Co-occurrence Matrix** (10 min)
   - Count word pairs within window
   - Build sparse matrix representation

2. **Dimensionality Reduction** (10 min)
   - Apply SVD to compress to dense vectors
   - Understand information preservation

3. **Similarity Search** (10 min)
   - Implement cosine similarity
   - Find similar words
   - Simple word analogies

### What You'll Build
```python
class SimpleEmbeddings:
    def __init__(self, dim=50):
        self.dim = dim
        self.word_to_idx = {}
        self.embeddings = None
    
    def train(self, corpus, window=2):
        # Build co-occurrence and reduce
        pass
    
    def get_vector(self, word) -> np.ndarray:
        # Return embedding for word
        pass
    
    def most_similar(self, word, top_k=5):
        # Find semantically similar words
        pass
    
    def analogy(self, a, b, c):
        # a is to b as c is to ?
        pass
```

### Key Takeaways
- ✅ Words with similar meanings have similar vectors
- ✅ Vector arithmetic captures relationships
- ✅ This is the foundation for all modern NLP

### Tech Stack
- Python 3.10+
- NumPy (matrix operations)
- Matplotlib (optional: visualization)

### Deliverables
- [ ] Starter code with co-occurrence skeleton
- [ ] Pre-trained mini embeddings for quick demos
- [ ] Slide deck with visualizations
- [ ] Take-home: Visualize embeddings with t-SNE

---

## Workshop 3: Vector Databases — Semantic Memory

### Learning Objectives
- Understand how semantic search differs from keyword search
- Build a simple vector store with indexing
- Implement approximate nearest neighbor search

### Session Breakdown

| Time | Activity |
|------|----------|
| 0-5 min | Hook: Search "happy dog" find "joyful puppy" |
| 5-20 min | Theory: Vector search fundamentals |
| 20-45 min | Build: In-memory vector database |
| 45-60 min | Q&A and wrap-up |

### Theory Concepts (~15 min)
- Keyword search vs. semantic search
- Curse of dimensionality
- Approximate Nearest Neighbor (ANN) algorithms
- HNSW, IVF, LSH overview (conceptual)

### Hands-On Build (~25-30 min)

**Build Progression:**
1. **Brute Force Search** (5 min)
   - Store vectors in list
   - Linear scan with cosine similarity

2. **Simple LSH Index** (15 min)
   - Random hyperplane hashing
   - Bucket-based retrieval
   - Trade accuracy for speed

3. **Document Store** (10 min)
   - Store text + embedding pairs
   - Semantic document search
   - Return ranked results

### What You'll Build
```python
class SimpleVectorDB:
    def __init__(self, dim, num_buckets=10):
        self.dim = dim
        self.documents = []
        self.vectors = []
        self.index = {}  # LSH buckets
    
    def add(self, text, vector):
        # Store document with its embedding
        pass
    
    def _hash(self, vector) -> str:
        # LSH hash function
        pass
    
    def search(self, query_vector, top_k=5):
        # Find similar documents
        pass
```

### Key Takeaways
- ✅ Vector DBs enable "search by meaning"
- ✅ Indexing trades accuracy for speed
- ✅ This is the "memory" component of RAG

### Tech Stack
- Python 3.10+
- NumPy
- Previous workshop's embeddings

### Deliverables
- [ ] Starter code with document store skeleton
- [ ] Sample document corpus (50-100 short texts)
- [ ] Slide deck with ANN visualizations
- [ ] Take-home: Compare with ChromaDB/FAISS

---

## Workshop 4: Attention — The Attention Revolution

### Learning Objectives
- Understand why attention replaced RNNs
- Implement self-attention from scratch
- Visualize attention patterns

### Session Breakdown

| Time | Activity |
|------|----------|
| 0-5 min | Hook: Attention heatmap on a sentence |
| 5-20 min | Theory: Query, Key, Value explained |
| 20-45 min | Build: Self-attention mechanism |
| 45-60 min | Q&A and wrap-up |

### Theory Concepts (~15 min)
- Problems with RNNs (sequential, vanishing gradients)
- Attention as "soft lookup"
- Query, Key, Value intuition
- Scaled dot-product attention
- Multi-head attention (conceptual)

### Hands-On Build (~25-30 min)

**Build Progression:**
1. **Attention Scores** (10 min)
   - Compute Q, K, V matrices
   - Dot product attention
   - Softmax normalization

2. **Scaled Attention** (5 min)
   - Add scaling factor
   - Understand why it matters

3. **Attention Visualization** (10 min)
   - Generate attention weights
   - Plot heatmap
   - Interpret what model "sees"

### What You'll Build
```python
class SelfAttention:
    def __init__(self, dim, head_dim):
        self.W_q = np.random.randn(dim, head_dim) * 0.1
        self.W_k = np.random.randn(dim, head_dim) * 0.1
        self.W_v = np.random.randn(dim, head_dim) * 0.1
    
    def forward(self, x):
        # x: (seq_len, dim)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        scores = Q @ K.T / np.sqrt(K.shape[-1])
        weights = softmax(scores)
        return weights @ V, weights
```

### Key Takeaways
- ✅ Attention lets models focus on relevant parts
- ✅ It enables parallel processing of sequences
- ✅ This is the core innovation behind transformers

### Tech Stack
- Python 3.10+
- NumPy
- Matplotlib (attention visualization)

### Deliverables
- [ ] Starter code with matrix operation hints
- [ ] Pre-computed embeddings for sample sentences
- [ ] Slide deck with animated attention flow
- [ ] Take-home: Multi-head attention extension

---

## Workshop 5: Transformers — The Transformer Decoded

### Learning Objectives
- Understand the full transformer architecture
- Build a minimal decoder-only transformer
- See how text generation works

### Session Breakdown

| Time | Activity |
|------|----------|
| 0-5 min | Hook: Generate text with tiny model |
| 5-20 min | Theory: Transformer architecture |
| 20-45 min | Build: Mini-GPT |
| 45-60 min | Q&A and wrap-up |

### Theory Concepts (~15 min)
- Encoder vs. Decoder vs. Encoder-Decoder
- Positional encoding (why order matters)
- Layer normalization
- Feed-forward networks in transformers
- Causal masking for generation

### Hands-On Build (~25-30 min)

**Build Progression:**
1. **Positional Encoding** (5 min)
   - Sinusoidal encoding
   - Add position information

2. **Transformer Block** (15 min)
   - Self-attention (from Workshop 4)
   - Feed-forward layer
   - Layer normalization
   - Residual connections

3. **Text Generation** (10 min)
   - Causal masking
   - Next token prediction
   - Greedy/sampling decode

### What You'll Build
```python
class MiniTransformer:
    def __init__(self, vocab_size, dim, num_layers):
        self.embedding = np.random.randn(vocab_size, dim) * 0.1
        self.pos_encoding = self._create_pos_encoding(512, dim)
        self.layers = [TransformerBlock(dim) for _ in range(num_layers)]
        self.output = np.random.randn(dim, vocab_size) * 0.1
    
    def forward(self, tokens):
        x = self.embedding[tokens] + self.pos_encoding[:len(tokens)]
        for layer in self.layers:
            x = layer(x)
        return x @ self.output
    
    def generate(self, prompt_tokens, max_new=20):
        # Autoregressive generation
        pass
```

### Key Takeaways
- ✅ Transformers are attention + position + feed-forward
- ✅ Generation is iterative next-token prediction
- ✅ Scale (parameters, data) is what makes GPT powerful

### Tech Stack
- Python 3.10+
- NumPy
- Workshop 1 tokenizer

### Deliverables
- [ ] Starter code with layer skeletons
- [ ] Tiny pre-trained weights for demo
- [ ] Slide deck with architecture diagrams
- [ ] Take-home: Temperature and top-k sampling

---

## Workshop 6: RAG — Bringing It All Together

### Learning Objectives
- Understand the RAG pipeline end-to-end
- Build a complete RAG system using previous components
- See how retrieval augments generation

### Session Breakdown

| Time | Activity |
|------|----------|
| 0-5 min | Hook: Ask question, get sourced answer |
| 5-20 min | Theory: RAG architecture |
| 20-45 min | Build: Complete RAG pipeline |
| 45-60 min | Q&A and series wrap-up |

### Theory Concepts (~15 min)
- Why RAG? (Hallucination, freshness, grounding)
- Retrieval: Finding relevant context
- Augmentation: Injecting into prompt
- Generation: Producing grounded response
- Chunking strategies
- Reranking (conceptual)

### Hands-On Build (~25-30 min)

**Build Progression:**
1. **Document Ingestion** (5 min)
   - Chunk documents
   - Embed chunks
   - Store in vector DB (Workshop 3)

2. **Retrieval Pipeline** (10 min)
   - Embed query
   - Search vector store
   - Retrieve top-k chunks

3. **Augmented Generation** (15 min)
   - Build prompt with context
   - Generate with transformer (Workshop 5)
   - Format response with sources

### What You'll Build
```python
class SimpleRAG:
    def __init__(self, tokenizer, embedder, vector_db, generator):
        self.tokenizer = tokenizer      # Workshop 1
        self.embedder = embedder        # Workshop 2
        self.vector_db = vector_db      # Workshop 3
        self.generator = generator      # Workshop 5
    
    def ingest(self, documents):
        for doc in documents:
            chunks = self._chunk(doc)
            for chunk in chunks:
                vec = self.embedder.embed(chunk)
                self.vector_db.add(chunk, vec)
    
    def query(self, question, top_k=3):
        # Retrieve relevant chunks
        q_vec = self.embedder.embed(question)
        contexts = self.vector_db.search(q_vec, top_k)
        
        # Build augmented prompt
        prompt = self._build_prompt(question, contexts)
        
        # Generate response
        response = self.generator.generate(prompt)
        return response, contexts
```

### Key Takeaways
- ✅ RAG grounds LLMs in real data
- ✅ All workshop components connect together
- ✅ You've built the core of a production RAG system!

### Tech Stack
- Python 3.10+
- All previous workshops' code
- Sample knowledge base (10-20 documents)

### Deliverables
- [ ] Starter code integrating all components
- [ ] Sample knowledge base
- [ ] Slide deck with full architecture
- [ ] Take-home: Add reranking or hybrid search

---

## Pre-Workshop Setup

### Required Installations
```bash
# Python environment
python -m venv genai-workshop
source genai-workshop/bin/activate  # or genai-workshop\Scripts\activate on Windows

# Core dependencies
pip install numpy matplotlib jupyter

# Optional (for comparisons/extensions)
pip install scikit-learn chromadb sentence-transformers
```

### Hardware Requirements
- Any laptop from last 5 years
- 4GB RAM minimum
- No GPU required (everything runs on CPU)

### Pre-Workshop Checklist
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] NumPy and Matplotlib installed
- [ ] Jupyter notebooks working
- [ ] Starter code downloaded

---

## Handling Different Skill Levels

### For Faster Learners
- Extension challenges in each workshop
- Optional "deep dive" sections
- Take-home exercises with increasing difficulty

### For Those Who Struggle
- Pair programming encouraged
- Reference implementations available
- Focus on concepts over perfect code
- "Checkpoint" files at each stage

### Backup Plans
- Pre-written code blocks ready to paste
- Common error solutions documented
- Cloud backup environment (Colab notebooks)

---

## Sample Data

### Workshop 1-2: Text Corpus
```
A small corpus of 1000-2000 sentences covering:
- General knowledge
- Technical content
- Conversational text
Suggested: Simple Wikipedia subset or curated news articles
```

### Workshop 3: Document Collection
```
50-100 short documents (1-2 paragraphs each):
- FAQ-style content
- Product descriptions
- Knowledge base articles
```

### Workshop 5-6: Tiny Training Set
```
Pre-prepared to show generation works
Participants won't train from scratch (too slow)
```

---

## Series Schedule Options

| Format | Schedule | Best For |
|--------|----------|----------|
| Weekly | 6 weeks, 1 session/week | Regular meetup groups |
| Intensive | 2 days, 3 sessions/day | Weekend workshops |
| Bi-weekly | 3 months, 1 session/2 weeks | Corporate training |

---

## Success Metrics

### Participant Outcomes
- Can explain tokenization, embeddings, attention to a colleague
- Has working code for each component
- Understands how RAG connects the pieces
- Comfortable reading transformer-related code

### Workshop Quality
- Completion rate >80%
- Code runs for >90% of participants
- Q&A engagement in each session
- Positive feedback on demystification goal

---

## Quick Reference Cards

Create a one-page reference for each concept:

| Workshop | Reference Card Content |
|----------|----------------------|
| 1 | Tokenization comparison chart |
| 2 | Embedding operations cheatsheet |
| 3 | Vector similarity formulas |
| 4 | Attention equation breakdown |
| 5 | Transformer block diagram |
| 6 | RAG pipeline flowchart |

---

## Next Steps for Organizer

1. [ ] Review and customize this plan
2. [ ] Create slide decks for each workshop
3. [ ] Prepare starter code repositories
4. [ ] Test all code on target Python version
5. [ ] Prepare sample datasets
6. [ ] Set up communication channel (Discord/Slack)
7. [ ] Create feedback collection mechanism
8. [ ] Schedule dry-run with co-facilitators

---

## Repository Structure

```
genai-self-build/
├── app.py                        # Unified demo (all workshops)
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
├── docs/
│   └── workshop-plan.md          # This document
├── workshops/
│   ├── 01-tokenization/          # ✅ Complete
│   │   ├── tokenizer.py          # Implementation
│   │   ├── app.py                # Streamlit demo
│   │   ├── test_tokenizer.py     # Tests
│   │   ├── slides/               # Presentation
│   │   ├── cheatsheet.md
│   │   ├── qna.md
│   │   └── README.md
│   ├── 02-embeddings/            # ✅ Complete
│   ├── 03-vector-databases/      # ✅ Complete
│   ├── 04-attention/             # ✅ Complete
│   ├── 05-transformers/          # ✅ Complete
│   └── 06-rag/                   # ✅ Complete
└── data/
    └── (sample data files)
```

---

## Status

All 6 workshops are **complete** with:
- ✅ Core Python implementation
- ✅ Interactive Streamlit demo
- ✅ Comprehensive test suite
- ✅ Marp presentation slides
- ✅ Cheatsheet and Q&A documents
- ✅ Unified demo combining all workshops

---

*Author: Michael Kennedy (michael.kennedy@analog.com)*
*Last Updated: December 2025*
*Version: 1.0*
