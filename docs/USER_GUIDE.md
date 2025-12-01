# 🛸 GenAI Self-Build: User Guide

Welcome to the GenAI Self-Build Workshop Series! This guide will help you get the most out of the learning experience.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Workshop Overview](#workshop-overview)
3. [Running the Demos](#running-the-demos)
4. [Learning Path](#learning-path)
5. [Tips for Success](#tips-for-success)
6. [Troubleshooting](#troubleshooting)
7. [Additional Resources](#additional-resources)

---

## 🚀 Getting Started

### Prerequisites

Before starting, make sure you have:

- **Python 3.9 or higher** installed
- **Basic Python knowledge** (functions, loops, lists, dictionaries)
- **Command line familiarity** (navigating folders, running commands)
- **No AI/ML experience required!**

### Installation

1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/yourusername/genai-self-build.git
   cd genai-self-build
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import numpy; import streamlit; print('Ready to go!')"
   ```

---

## 📚 Workshop Overview

### The 6 Workshops

| # | Workshop | What You'll Learn | Time |
|---|----------|-------------------|------|
| 1 | **Tokenization** | How text becomes numbers | 1 hour |
| 2 | **Embeddings** | How meaning becomes vectors | 1 hour |
| 3 | **Vector Databases** | How to search by meaning | 1 hour |
| 4 | **Attention** | How models focus on what matters | 1 hour |
| 5 | **Transformers** | How GPT-style models work | 1 hour |
| 6 | **RAG** | How to ground AI in real data | 1 hour |

### The Alien Analogy 🛸

Throughout the series, we use a fun analogy: **an alien learning to understand human language**.

| Workshop | Alien's Journey | Screenshot |
|----------|-----------------|------------|
| 1 | 🛸 Building a codebook to read symbols | ![](pictures/01-tokenization.png) |
| 2 | 🗺️ Creating a map of where meanings live | ![](pictures/02-embeddings.png) |
| 3 | 📚 Organizing a magic library | ![](pictures/03-vector-db.png) |
| 4 | 👀 Learning what to focus on | ![](pictures/04-attention.png) |
| 5 | 🧠 Building a complete brain | ![](pictures/05-transformers.png) |
| 6 | 🔍 Getting a search engine | ![](pictures/06-rag.png) |

---

## 🎮 Running the Demos

### Option 1: Unified Demo (Recommended for Overview)

Run the combined demo that covers all workshops:

```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

### Option 2: Individual Workshop Demos

Each workshop has its own interactive demo:

```bash
# Workshop 1: Tokenization
cd workshops/01-tokenization
streamlit run app.py

# Workshop 2: Embeddings
cd workshops/02-embeddings
streamlit run app.py

# Workshop 3: Vector Databases
cd workshops/03-vector-databases
streamlit run app.py

# Workshop 4: Attention
cd workshops/04-attention
streamlit run app.py

# Workshop 5: Transformers
cd workshops/05-transformers
streamlit run app.py

# Workshop 6: RAG
cd workshops/06-rag
streamlit run app.py
```

### Option 3: Run the Core Code

Each workshop's main implementation can be run directly:

```bash
cd workshops/01-tokenization
python tokenizer.py  # Runs built-in tests
```

---

## 🎯 Learning Path

### Recommended Order

**Follow the workshops in order** — each builds on the previous:

```
1. Tokenization  →  2. Embeddings  →  3. Vector DB
                                            ↓
6. RAG  ←  5. Transformers  ←  4. Attention
```

### For Each Workshop

1. **Read the README.md** in the workshop folder
2. **Review the cheatsheet.md** for quick reference
3. **Run the interactive demo** (`streamlit run app.py`)
4. **Explore the code** in the main `.py` file
5. **Run the tests** to verify understanding
6. **Check qna.md** for common questions

### Workshop Materials

Each workshop folder contains:

| File | Purpose |
|------|---------|
| `*.py` | Core implementation with detailed comments |
| `app.py` | Interactive Streamlit demo |
| `test_*.py` | Test suite to verify functionality |
| `README.md` | Workshop overview and concepts |
| `cheatsheet.md` | 1-2 page quick reference |
| `qna.md` | Anticipated questions & answers |
| `slides/` | Presentation materials |

---

## 💡 Tips for Success

### Before Each Workshop

- [ ] Review the workshop README
- [ ] Glance at the cheatsheet
- [ ] Have the code open in your editor

### During the Workshop

- **Don't just copy-paste** — type the code yourself
- **Experiment** — change values and see what happens
- **Ask questions** — there are no dumb questions
- **Take notes** — jot down "aha!" moments

### After Each Workshop

- [ ] Run the test suite to verify understanding
- [ ] Try the "Take-Home Exercises" in the README
- [ ] Explain the concept to someone else (rubber duck debugging!)
- [ ] Connect it to the bigger picture

### Understanding the Code

The code is written to be **educational, not production-ready**:

- **Verbose comments** explain the "why"
- **Simple implementations** over optimized ones
- **NumPy only** — no hidden magic from ML libraries
- **Small scale** — same concepts, manageable size

---

## 🔧 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'streamlit'"

```bash
pip install streamlit
```

#### "ModuleNotFoundError: No module named 'numpy'"

```bash
pip install numpy
```

#### Streamlit won't start / port in use

```bash
# Try a different port
streamlit run app.py --server.port 8502
```

#### Import errors between workshops

Make sure you're running from the correct directory:

```bash
cd workshops/01-tokenization
python tokenizer.py  # ✅ Correct

python workshops/01-tokenization/tokenizer.py  # ❌ May have import issues
```

#### Tests failing

- Check you're using Python 3.9+
- Ensure numpy is installed
- Read the error message — it usually tells you what's wrong

### Getting Help

1. **Check the qna.md** file in each workshop
2. **Review the error message** carefully
3. **Google the error** — someone else has probably hit it
4. **Contact:** michael.kennedy@analog.com

---

## 📖 Additional Resources

### Recommended Reading

After completing the workshops, explore these resources:

- **"Attention Is All You Need"** — The original transformer paper
- **"The Illustrated Transformer"** — Jay Alammar's visual guide
- **Andrej Karpathy's YouTube** — "Let's build GPT from scratch"
- **Hugging Face Course** — Practical NLP with transformers

### Tools to Explore

- **tiktoken** — OpenAI's production tokenizer
- **sentence-transformers** — Pre-trained embeddings
- **FAISS / ChromaDB** — Production vector databases
- **Hugging Face Transformers** — Pre-trained models

### Next Steps

After completing all 6 workshops:

1. **Build something!** — A simple chatbot, search engine, or Q&A system
2. **Try real models** — Use Hugging Face to load pre-trained transformers
3. **Scale up** — Apply concepts with production tools
4. **Teach others** — Best way to solidify understanding

---

## 🎉 Congratulations!

By completing this series, you'll understand:

- ✅ How text becomes numbers (tokenization)
- ✅ How meaning becomes vectors (embeddings)
- ✅ How to search by meaning (vector databases)
- ✅ How models focus on relevant information (attention)
- ✅ How GPT-style models generate text (transformers)
- ✅ How to ground AI in real documents (RAG)

**You've demystified GenAI!** 🛸

---

*Questions? Contact: michael.kennedy@analog.com*
