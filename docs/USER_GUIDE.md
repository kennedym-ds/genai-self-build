# 🛸 GenAI Self-Build: User Guide

Welcome aboard, explorer! This guide helps you navigate the GenAI Self-Build Workshop Series — from a zero-install browser experience to full Python deep dives — while following Zara’s alien journey from confusion to understanding.

---

## 📋 Table of Contents

1. Getting Started
2. Workshop Overview
3. Running the Demos
4. Learning Path
5. Tips for Success
6. Troubleshooting
7. Additional Resources

---

## 🚀 Getting Started

### Start Here (Recommended): Interactive HTML Webpage

The easiest way to begin is the interactive browser app:

👉 **Just open `index.html` in your browser — that's it!**

No Python install. No environment setup. No package troubleshooting.

Inside the webpage, you can choose two learning styles:

- **Text Path** — language-based examples and explanations
- **Visual Path** — shape/layout-based intuition for the same core concepts

Both paths follow Zara’s chapter-by-chapter story and teach the same six GenAI building blocks.

### Optional Setup for Python Deep Dives

If you want to run the workshop code, Streamlit demos, and tests locally, set up Python:

#### Prerequisites

Before starting, make sure you have:

- **Python 3.9 or higher** installed
- **Basic Python knowledge** (functions, loops, lists, dictionaries)
- **Command line familiarity** (navigating folders, running commands)
- **No AI/ML experience required!**

#### Installation

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

|#|Workshop|What You'll Learn|Tests|
|---|---|---|---|
|1|**Tokenization**|How text becomes numbers|19|
|2|**Embeddings**|How meaning becomes vectors|21|
|3|**Vector Databases**|How to search by meaning|22|
|4|**Attention**|How models focus on what matters|21|
|5|**Transformers**|How GPT-style models work|25|
|6|**RAG**|How to ground AI in real data|25|
|✅|**Total**|**All workshop suites combined**|**133**|

### The Alien Analogy 🛸

Throughout the series, we use a fun analogy: **an alien learning to understand human language**.

|Workshop|Alien's Journey|Screenshot|
|---|---|---|
|1|🛸 Building a codebook to read symbols|![Tokenization screenshot](pictures/01-tokenization.png)|
|2|🗺️ Creating a map of where meanings live|![Embeddings screenshot](pictures/02-embeddings.png)|
|3|📚 Organizing a magic library|![Vector DB screenshot](pictures/03-vector-db.png)|
|4|👀 Learning what to focus on|![Attention screenshot](pictures/04-attention.png)|
|5|🧠 Building a complete brain|![Transformers screenshot](pictures/05-transformers.png)|
|6|🔍 Getting a search engine|![RAG screenshot](pictures/06-rag.png)|

---

## 🎮 Running the Demos

### Option A: Interactive HTML Webpage (Recommended)

This is the fastest and most learner-friendly path:

👉 **Just open `index.html` in your browser — that's it!**

What you get:

- A **chapter-based journey** through Zara’s story
- **Text** and **Visual** learning paths for each concept
- Navigation across Home → Chapters 1-6 → Finale
- Zero-install, instant exploration

### Option B: Deep Dive with Python (Individual Workshops)

Each workshop has its own Streamlit app, implementation, and test suite.

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

You can also run each workshop’s core implementation and tests directly:

```bash
cd workshops/01-tokenization
python tokenizer.py
python test_tokenizer.py
```

### Option C: Unified Streamlit Demo

Run the combined root app that covers all six workshops in one place:

```bash
streamlit run app.py
```

Then open your browser to: **[http://localhost:8501](http://localhost:8501)**

---

## 🎯 Learning Path

### Recommended Order

**Follow the workshops in order** — each builds on the previous:

```text
1. Tokenization  →  2. Embeddings  →  3. Vector DB
                                            ↓
6. RAG  ←  5. Transformers  ←  4. Attention
```

### Suggested Flow (Best Experience)

1. Start in the HTML app (`index.html`)
2. Use **chapter-by-chapter navigation** to follow Zara’s full arc
3. Toggle between **Text** and **Visual** paths to reinforce understanding
4. Move to **Option B** for code-level exploration and tests
5. Use **Option C** when you want one unified Streamlit walkthrough

### For Each Workshop (Python Deep Dive)

1. **Read the `README.md`** in the workshop folder
2. **Review the `cheatsheet.md`** for quick reference
3. **Run the interactive demo** (`streamlit run app.py`)
4. **Explore the code** in the main `.py` file
5. **Run the tests** to verify understanding
6. **Check `qna.md`** for common questions

### Workshop Materials

Each workshop folder contains:

|File|Purpose|
|---|---|
|`*.py`|Core implementation with detailed comments|
|`app.py`|Interactive Streamlit demo|
|`test_*.py`|Test suite to verify functionality|
|`README.md`|Workshop overview and concepts|
|`cheatsheet.md`|1-2 page quick reference|
|`qna.md`|Anticipated questions & answers|
|`slides/`|Presentation materials|

---

## 💡 Tips for Success

### Before Each Workshop

- [ ] Decide your mode: **HTML quick learning** or **Python deep dive**
- [ ] Review the workshop README
- [ ] Glance at the cheatsheet
- [ ] Have the code open in your editor (if doing Option B/C)

### During the Workshop

- **Don't just copy-paste** — type the code yourself
- **Experiment** — change values and see what happens
- **Switch Text/Visual modes** in the HTML app to build intuition from both angles
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
- **NumPy + Streamlit focus** — no hidden magic from large frameworks
- **Small scale** — same concepts, manageable size

---

## 🔧 Troubleshooting

### First, the easy win ✅

If you use the HTML app (`index.html`), there is **no install required** — so there is usually **nothing to troubleshoot**.

### Common Python/Streamlit Issues (Option B/C)

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

1. **Check the `qna.md`** file in each workshop
2. **Review the error message** carefully
3. **Google the error** — someone else has probably hit it
4. **Contact:** [michael.kennedy@analog.com](mailto:michael.kennedy@analog.com)

---

## 📖 Additional Resources

### For Presenters 🎤

If you're presenting this material live, use the keynote package:

- `keynote/app.py` — presenter-focused Streamlit demo
- `keynote/SCRIPT.md` — full talk track/script
- `keynote/html_app/` — browser-friendly keynote assets

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

## 🎉 Congratulations

By completing this series, you'll understand:

- ✅ How text becomes numbers (tokenization)
- ✅ How meaning becomes vectors (embeddings)
- ✅ How to search by meaning (vector databases)
- ✅ How models focus on relevant information (attention)
- ✅ How GPT-style models generate text (transformers)
- ✅ How to ground AI in real documents (RAG)

**You've demystified GenAI!** 🛸

---

*Questions? Contact: [michael.kennedy@analog.com](mailto:michael.kennedy@analog.com)*
