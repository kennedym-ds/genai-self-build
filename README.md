# 🛸 GenAI Self-Build Workshop Series

**Demystifying Generative AI Through Hands-On Construction**

> "What I cannot create, I do not understand." — Richard Feynman

## What Is This?

A 6-part workshop series where participants build GenAI components **from scratch**. No magic, no black boxes — just Python and understanding.

Each workshop is **1 hour** (45 min teaching/demo + 15 min Q&A) and includes:
- Complete Python implementation **with debug mode**
- Interactive Streamlit demo
- Comprehensive test suite
- Presentation slides
- Cheatsheet and Q&A guide
- **🔍 "Under the Hood" documentation** revealing how it really works

## 🎯 The Workshops

| # | Topic | Analogy | What You'll Build |
|---|-------|---------|-------------------|
| 1 | **Tokenization** | 🛸 Alien builds a codebook | Character, word, and BPE tokenizers |
| 2 | **Embeddings** | 🗺️ Creating a meaning map | Word vectors from co-occurrence |
| 3 | **Vector Database** | 📚 Magic library shelves | In-memory vector store with similarity search |
| 4 | **Attention** | 👀 Spotlight of focus | Self-attention with visualization |
| 5 | **Transformers** | 🧠 Complete alien brain | Mini decoder-only transformer |
| 6 | **RAG** | 🔍 Search engine + brain | Complete retrieval-augmented generation |

## 📖 The Story: Zara's Journey

The workshops are woven together by a **narrative thread** — the journey of **Zara**, a Zorathian scientist who lands on Earth and must learn to understand human language from scratch.

Each workshop is a chapter in her story:

| Chapter | Zara's Challenge | What She Builds |
|---------|-----------------|-----------------|
| 📕 The Codebook | Can't read human text | A tokenizer to turn symbols → numbers |
| 📗 The Map | Numbers don't capture meaning | An embedding space where similar = nearby |
| 📘 The Library | Too many vectors to search | A vector database with instant lookup |
| 📙 The Spotlight | Context keeps confusing her | An attention mechanism to focus |
| 📓 The Brain | Needs to assemble everything | A complete transformer architecture |
| 📔 The Search Engine | Her brain hallucinates | RAG — look things up before answering |

> *By the end, Zara has built every component of a modern AI system — and so have you.*

See **[📖 Story Guide](docs/STORY_GUIDE.md)** for the complete narrative with presenter scripts, transition lines, and emotional arc guidance.

## 📷 Screenshots

<p align="center">
  <img src="docs/pictures/unified-demo.png" alt="Unified Demo" width="800"/>
</p>

<details>
<summary>📸 View All Workshop Screenshots</summary>

| Workshop | Screenshot |
|----------|------------|
| 1. Tokenization | ![Tokenization](docs/pictures/01-tokenization.png) |
| 2. Embeddings | ![Embeddings](docs/pictures/02-embeddings.png) |
| 3. Vector DB | ![Vector DB](docs/pictures/03-vector-db.png) |
| 4. Attention | ![Attention](docs/pictures/04-attention.png) |
| 5. Transformers | ![Transformers](docs/pictures/05-transformers.png) |
| 6. RAG | ![RAG](docs/pictures/06-rag.png) |

</details>

## 🚀 Quick Start

### Option A: Interactive Webpage (Zero Install)

Open `index.html` in any modern browser — no server, no dependencies. Includes a Text/Visual dual learning path covering all 6 concepts.

### Option B: Python Workshops + Streamlit

```bash
# Clone the repository
git clone https://github.com/yourusername/genai-self-build.git
cd genai-self-build

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

**Run the Unified Streamlit Demo:**

```bash
streamlit run app.py
```

**Run Individual Workshops:**

```bash
cd workshops/01-tokenization
streamlit run app.py       # Interactive demo
python test_tokenizer.py   # Run tests
```

### Option C: Keynote Demo (Presenter Mode)

```bash
cd keynote
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
genai-self-build/
├── index.html                # 🌐 Interactive webpage (Text + Visual paths)
├── css/index.css             # Webpage styles
├── js/                       # Webpage JavaScript
│   ├── app.js                # SPA router & state
│   ├── ml/                   # ML engines (tokenizer, embeddings, vector DB, attention)
│   └── ui/                   # UI components & page renderers
│
├── app.py                    # 🎮 Unified Streamlit demo (all workshops)
├── requirements.txt          # Python dependencies
│
├── workshops/                # 📚 Python workshop implementations
│   ├── 01-tokenization/      # 🔤 Text to numbers (19 tests)
│   │   ├── tokenizer.py      # Implementation
│   │   ├── app.py            # Streamlit demo
│   │   ├── test_tokenizer.py # Test suite
│   │   ├── slides/           # Presentation
│   │   ├── cheatsheet.md     # Quick reference
│   │   ├── qna.md            # Q&A guide
│   │   └── README.md         # Workshop docs
│   │
│   ├── 02-embeddings/        # 🗺️ Meaning in vectors (21 tests)
│   ├── 03-vector-databases/  # 📚 Semantic search (22 tests)
│   ├── 04-attention/         # 👀 Focus mechanism (21 tests)
│   ├── 05-transformers/      # 🧠 The architecture (25 tests)
│   └── 06-rag/               # 🔍 Retrieval + generation (25 tests)
│
├── keynote/                  # 🎤 Keynote presentation demo
│   ├── app.py                # Streamlit keynote app
│   ├── SCRIPT.md             # TED-style 20-min script
│   ├── slides/               # Marp slide deck
│   └── html_app/             # Static HTML version
│
├── shape_data.js             # Visual-path shape definitions
├── shapes_core.js            # Visual-path ML engines
│
└── docs/                     # 📖 Documentation
    ├── STORY_GUIDE.md        # Zara narrative framework
    ├── USER_GUIDE.md         # Learner guide
    ├── TEACHER_GUIDE.md      # Facilitator guide
    ├── UNDER_THE_HOOD.md     # Deep-dive algorithm docs
    └── workshop-plan.md      # Curriculum guide
```

## 📋 Workshop Contents

Each workshop folder contains:

| File | Description |
|------|-------------|
| `*.py` | Core implementation with detailed comments |
| `app.py` | Interactive Streamlit demo |
| `test_*.py` | Comprehensive test suite |
| `slides/slides.md` | Marp presentation |
| `cheatsheet.md` | 1-2 page quick reference |
| `qna.md` | Anticipated questions & answers |
| `README.md` | Workshop documentation |

**Total: 133 tests across 6 suites** (19 + 21 + 22 + 21 + 25 + 25)

### Interactive Webpage

The `index.html` single-page app provides a parallel learning experience with:
- **Text Path** — Type natural language and see tokenization, embeddings, vector search, attention, and generation in real time
- **Visual Path** — Learn the same concepts through Zara's shape-based world (circles, squares, triangles instead of words)

## 🔗 How It All Connects

```
┌─────────────────────────────────────────────────────────────────┐
│                    The GenAI Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📝 Text  →  🔤 Tokens  →  📊 Embeddings  →  🔍 Search         │
│  (input)    (Workshop 1)   (Workshop 2)     (Workshop 3)       │
│                                                                 │
│                    ↓                                            │
│                                                                 │
│  💬 Answer  ←  🧠 Transform  ←  👀 Attend  ←  📄 Context       │
│  (output)     (Workshop 5)    (Workshop 4)   (Workshop 6)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🎓 Prerequisites

- **Python:** Basic knowledge (functions, loops, lists, dictionaries)
- **Command line:** Basic familiarity
- **AI/ML:** No experience required!

## 📚 Dependencies

Core dependencies (minimal by design):
- `numpy>=1.21.0` — Numerical operations
- `streamlit>=1.28.0` — Interactive demos

## 🎯 Philosophy

We build everything from scratch (or near-scratch) using NumPy:
- **No hidden library calls** — Every operation is visible
- **Simplified but real** — Same algorithms, smaller scale
- **Learn by doing** — Build it yourself to understand it

### 🔍 Under the Hood Features

**NEW!** All implementations include debug mode for deep learning:

```python
from tokenizer import SimpleTokenizer

# Enable debug mode to see internals
tokenizer = SimpleTokenizer(strategy='word', debug=True)
tokenizer.train(corpus)  # Shows step-by-step training process

# Get detailed statistics
stats = tokenizer.get_stats()
print(f"Vocabulary covers {stats['unique_words']} words")
print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
```

**What you'll see:**
- ✅ Step-by-step algorithm execution
- ✅ Intermediate data transformations
- ✅ Performance metrics and statistics
- ✅ Decision points and why they're made
- ✅ Comparison to production systems

**Try it:**
```bash
cd workshops/01-tokenization
python under_the_hood_demo.py  # Interactive demo with debug mode
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **[📖 Story Guide](docs/STORY_GUIDE.md)** | Narrative framework with presenter scripts and transition lines |
| **[User Guide](docs/USER_GUIDE.md)** | Getting started, running demos, learning tips |
| **[Teacher Guide](docs/TEACHER_GUIDE.md)** | Facilitation tips, session timelines, handling Q&A |
| **[Workshop Plan](docs/workshop-plan.md)** | Complete curriculum and session details |
| **[🔍 Under the Hood](docs/UNDER_THE_HOOD.md)** | Deep dive into algorithms, comparisons to production systems |
| **[🎤 Keynote Script](keynote/SCRIPT.md)** | TED-style 20-minute presentation script |

## 📖 Additional Resources

After completing the workshops:
- **"Attention Is All You Need"** — The original transformer paper
- **"The Illustrated Transformer"** — Jay Alammar's visual guide
- **Andrej Karpathy's YouTube** — "Let's build GPT from scratch"
- **Hugging Face Course** — Practical NLP with transformers

## 📄 License

MIT — Use freely for education and learning.

## 👤 Author

**Michael Kennedy**  
📧 michael.kennedy@analog.com

---

<div align="center">

**🛸 Demystifying AI, one concept at a time**

*Built to make AI understandable*

</div>
