# GenAI Self-Build Workshop Series - Agent Instructions

## Project Overview

A 6-part workshop series to **demystify GenAI concepts** through hands-on, locally-run implementations. Each workshop is 1 hour (45 min teaching/demo + 15 min Q&A).

**Author:** Michael Kennedy (michael.kennedy@analog.com)

### Workshop Structure
```
workshops/
├── 01-tokenization/     ✅ Complete
├── 02-embeddings/       ✅ Complete
├── 03-vector-databases/ ✅ Complete
├── 04-attention/        ✅ Complete
├── 05-transformers/     ✅ Complete
└── 06-rag/              ✅ Complete
```

**Unified Demo:** `app.py` in root directory combines all workshops.

---

## Style Guide

### 🛸 Core Analogy Pattern
Each workshop MUST have a central analogy that runs throughout ALL materials. Use the **alien learning** theme as a connecting thread:

| Workshop | Core Analogy | Theme |
|----------|--------------|-------|
| 1. Tokenization | 🛸 Alien building a codebook | Learning to read symbols |
| 2. Embeddings | 🗺️ Alien creating a map of meaning | Where words live in space |
| 3. Vector DB | 📚 Alien's library with magic shelves | Finding similar things fast |
| 4. Attention | 👀 Alien learning what to focus on | Context matters |
| 5. Transformers | 🧠 Alien's full brain architecture | Putting it all together |
| 6. RAG | 🔍 Alien with a search engine | Knowledge retrieval |

### Supporting Analogies Per Workshop
Include 3-4 additional relatable analogies:
- 📞 Phone/communication analogies
- 📖 Book/library analogies  
- 📦 Shipping/packaging analogies
- 🍳 Cooking/recipe analogies
- 🧩 Puzzle/jigsaw analogies

---

## File Structure Per Workshop

Each workshop folder MUST contain:

```
workshops/XX-topic/
├── topic.py              # Main implementation (well-commented)
├── app.py                # Streamlit interactive demo
├── test_topic.py         # Comprehensive test suite
├── requirements.txt      # Dependencies
├── README.md             # Overview with Mermaid diagrams
├── cheatsheet.md         # Quick reference (1-2 pages)
├── qna.md                # Anticipated Q&A
└── slides/
    └── slides.md         # Marp-compatible presentation
```

---

## Code Style

### Python File Header
```python
"""
🎯 Workshop X: Title Here
=========================

🛸 THE [ANALOGY] ANALOGY:
[2-3 paragraph explanation using the core analogy]

THIS IS EXACTLY WHAT [CONCEPT] DOES.
[1 sentence connecting analogy to technical concept]

[Brief technical description]

Usage:
    python filename.py
"""
```

### Class Documentation
```python
class ClassName:
    """
    Brief description of what this class does.
    
    Example usage:
        instance = ClassName()
        instance.method("input")  # Expected output
    """
```

### Method Sections
Use clear section headers with emoji:
```python
# =========================================================================
# PART 1: SECTION NAME (Start here!)
# =========================================================================
```

### Inline Comments
- Explain the "why" not just the "what"
- Use step-by-step numbering for algorithms
- Include example inputs/outputs in docstrings

---

## Streamlit App Pattern

### Structure
```python
# 1. Page config
st.set_page_config(title, icon, layout="wide")

# 2. Custom CSS for visualizations
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# 3. Title with analogy tagline
st.title("🛸 Topic Explorer")
st.markdown("### *Analogy-based subtitle*")

# 4. Tabs for organization
tab1, tab2, tab3 = st.tabs(["🔬 Interactive Demo", "📊 How It Works", "🆚 Compare"])

# 5. Sidebar with:
#    - Analogy explanation
#    - Settings
#    - Workshop number indicator

# 6. Visual data flow diagram at top of main tab

# 7. Interactive controls with immediate feedback

# 8. Comparison section showing tradeoffs

# 9. Footer with workshop branding
```

### Visual Elements Required
- **Data flow pipeline** showing input → process → output
- **Color-coded tokens/elements** with consistent palette
- **Metrics** showing key numbers (vocab size, dimensions, etc.)
- **Comparison tables** highlighting tradeoffs
- **Error states** with red highlighting for failures

### Color Palette
```css
.token-char { background-color: #3b82f6; }  /* Blue */
.token-word { background-color: #8b5cf6; }  /* Purple */
.token-bpe { background-color: #10b981; }   /* Green */
.error { background-color: #dc2626; }        /* Red */
.success { background-color: #16a34a; }      /* Green */
.warning { background-color: #f59e0b; }      /* Amber */
```

---

## Slide Deck Pattern (Marp)

### Header
```markdown
---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #ffffff
---
```

### Slide Structure
1. **Title slide** with emoji and analogy teaser
2. **Learning objectives** (3-4 bullet points)
3. **Analogy introduction** with visual
4. **Concept explanation** with Mermaid diagram
5. **Code walkthrough** (3-4 slides)
6. **Live demo** slide
7. **Comparison/tradeoffs** slide
8. **Key takeaways** (3 points max)
9. **Preview of next workshop**
10. **Q&A** slide

### Mermaid Diagrams
Include at least 2 diagrams per deck:
- Flow diagram showing data transformation
- Comparison diagram showing approaches

---

## Test Suite Pattern

### Structure
```python
class TestResult:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

def run_tests():
    """Run all tests with emoji feedback."""
    results = TestResult()
    
    # Group 1: Basic functionality
    print("\n📦 GROUP 1: Basic Tests")
    print("-" * 40)
    
    # Individual tests with try/except
    try:
        # Test code
        results.passed += 1
        print("  ✅ Test name")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Test name: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 RESULTS: {results.passed} passed, {results.failed} failed")
```

### Test Categories
1. **Initialization tests** - Constructor, defaults
2. **Core functionality tests** - Main methods
3. **Edge case tests** - Empty input, special chars
4. **Integration tests** - End-to-end workflows

---

## README Pattern

### Structure
```markdown
# 🎯 Workshop X: Title

## 🛸 The Analogy
[Core analogy explanation]

## 📋 What You'll Learn
- Point 1
- Point 2
- Point 3

## 🏗️ Architecture
[Mermaid diagram]

## 🚀 Quick Start
[Code example]

## 📁 Files
[File descriptions]

## 🔗 Connection to LLMs
[How this relates to production systems]

## ➡️ Next Workshop
[Preview of next topic]
```

---

## Q&A Document Pattern

Structure with categories:
1. **Conceptual Questions** - "Why does X work this way?"
2. **Technical Questions** - "How do I implement Y?"
3. **Comparison Questions** - "What's the difference between A and B?"
4. **Real-World Questions** - "How does ChatGPT do this?"

Each answer should:
- Start with the analogy
- Give the technical explanation
- Provide a concrete example

---

## Cheatsheet Pattern

One-page reference with:
- **Quick definitions** in a table
- **Code snippets** for common operations
- **Mermaid diagram** of the core concept
- **Gotchas/tips** section
- **"Remember"** callout box with key insight

---

## Workshop Connections

Each workshop should:
1. **Reference previous workshop** in the intro
2. **Show how it builds** on previous concepts
3. **Preview next workshop** at the end
4. **Use consistent variable names** across workshops

### Shared Concepts
```python
# Use these consistently:
corpus = [...]           # Training data
vocab = {}               # Token/word mappings
encode(text) -> [ids]    # Text to numbers
decode(ids) -> text      # Numbers to text
```

---

## Dependencies

Keep minimal and standard:
```
numpy>=1.21.0
streamlit>=1.28.0
```

Add workshop-specific only when needed:
- Workshop 3+: `faiss-cpu` or `chromadb`
- Workshop 5+: `torch` (if needed)

---

## Quality Checklist

Before completing a workshop, verify:

- [ ] Core analogy runs through ALL materials
- [ ] Code has comprehensive comments
- [ ] Streamlit app has visual data flow
- [ ] Streamlit app shows clear tradeoffs/comparisons
- [ ] Test suite has 15+ tests, all passing
- [ ] Slides have 2+ Mermaid diagrams
- [ ] README connects to real LLM systems
- [ ] Cheatsheet fits on 1-2 pages
- [ ] Q&A anticipates "but how does ChatGPT..." questions
- [ ] Footer/headers show workshop number (X of 6)

---

## Example: Workshop 1 Reference

See `workshops/01-tokenization/` for the gold standard implementation:
- `tokenizer.py` - Three strategies with alien analogy
- `app.py` - Interactive demo with comparison tab
- `test_tokenizer.py` - 19 tests across 4 categories
- Complete slides, README, cheatsheet, Q&A
