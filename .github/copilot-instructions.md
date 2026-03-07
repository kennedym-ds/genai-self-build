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

**Unified Streamlit Demo:** `app.py` in root directory combines all workshops (Python path).

**Interactive Web App:** `index.html` in root — zero-install HTML SPA with dual Text/Visual learning paths and chapter-based navigation (Zara's journey).

**Keynote Demo:** `keynote/` — 20-minute TED-style presentation with companion Streamlit app and HTML app.

### Dual-Stack Architecture
```
Root (HTML SPA)             Root (Python Deep Dive)
├── index.html              ├── app.py (Streamlit unified)
├── css/index.css           └── workshops/01-06/
├── js/                         ├── topic.py
│   ├── app.js                  ├── app.py (Streamlit per-workshop)
│   ├── ml/                     ├── test_topic.py
│   │   ├── tokenizer.js        └── slides/slides.md
│   │   ├── embeddings.js
│   │   ├── vector_db.js    Keynote
│   │   ├── attention.js    ├── keynote/app.py
│   │   └── text_utils.js   ├── keynote/SCRIPT.md
│   └── ui/                 └── keynote/html_app/
│       ├── sidebar.js
│       └── pages/*.js
├── shape_data.js
└── shapes_core.js
```

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

## HTML SPA Pattern (index.html)

### Architecture
The HTML SPA uses a hash-based router with global state management:

```javascript
// AppState — global reactive state
const AppState = {
    learningPath: "Text",    // "Text" | "Visual"
    activeRoute: "home",
    engines: { tokenizer, embedding, db, attention }
};

// Router — hash-based navigation
const Router = {
    navigate(route, force = false) { ... }
};

// Each page exports: { render(), attachEvents() }
const MyPage = {
    render() { return '<div>...</div>'; },
    attachEvents() { /* bind handlers after render */ }
};
```

### Page Module Pattern
Each page in `js/ui/pages/` follows:
```javascript
const TopicPage = {
    render() {
        // 1. Branch on AppState.learningPath ("Text" / "Visual")
        // 2. Return HTML string with interactive controls
        // 3. Include chapterNav(routeName) at bottom for story navigation
        return `<div class="page">...</div>`;
    },
    attachEvents() {
        // Bind click/input handlers AFTER render
        // Use escapeHtml() for ALL user input displayed in DOM
    }
};
```

### Security Requirements
- **Always** use `escapeHtml()` (defined in `js/app.js`) for user-provided text before innerHTML
- **Never** use `eval()` or inject raw user input into HTML
- Shape engine outputs do NOT need escaping (internal data only)

### Shared Utilities
`js/ml/text_utils.js` provides `TextUtils` with shared functions:
- `TextUtils.hashStr(str)` — deterministic string hash
- `TextUtils.makeVector(text, dim)` — text to vector (demo-quality)
- `TextUtils.cosSim(a, b)` — cosine similarity
- `TextUtils.renderSceneHtml(scene, scale, height)` — SVG scene renderer

### Shape Engine APIs (Visual Path)
```javascript
// Shape engines in shapes_core.js + shape_data.js
const tokenizer = new ShapeTokenizer('part');
tokenizer.train(scenes);
tokenizer.encode(scene);  // → [token_ids]
tokenizer.decode(ids);     // → [shape_parts]

const embedding = new ShapeEmbedding('spatial', 16);
embedding.train(scenes);
embedding.get_embedding(sceneName);  // → Float64Array

const db = new ShapeVectorDB(16);
db.add_batch(names, embeddings);
db.search(queryVector, k);  // → [{name, score}]

const attention = new ShapeAttention();
attention.compute_attention(scene);  // → [weights, labels]
```

### Chapter Navigation System
The app uses a narrative chapter system defined in `js/app.js`:
- `CHAPTERS` array — maps routes to story chapter titles
- `CHAPTER_TRANSITIONS` — cliffhanger strings between chapters
- `chapterNav(currentRoute)` — generates prev/next navigation + story transition HTML

### CSS Theme
Dark indigo-violet theme using CSS custom properties in `css/index.css`:
```css
--bg-primary: #0f0a1a;
--text-primary: #e0d6f0;
--accent-primary: #a78bfa;    /* Violet */
--accent-secondary: #818cf8;   /* Indigo */
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

### HTML SPA
Zero dependencies — vanilla HTML/CSS/JS. Just open `index.html`.

### Python Workshops
Keep minimal and standard:
```
numpy>=1.21.0
streamlit>=1.28.0
```

Add workshop-specific only when needed:
- Workshop 3+: `faiss-cpu` or `chromadb`
- Workshop 5+: `torch` (if needed)

### Keynote
```
streamlit>=1.28.0
python-pptx>=0.6.21
```

---

## Quality Checklist

Before completing a workshop, verify:

### Python Workshop
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

### HTML SPA Page
- [ ] Supports BOTH Text and Visual learning paths
- [ ] Uses `escapeHtml()` for all user input in innerHTML
- [ ] Uses `TextUtils` for shared functions (no copy-paste)
- [ ] Includes `chapterNav()` at bottom for story flow
- [ ] Visual path uses AppState.engines correctly
- [ ] Zara narrative is consistent with STORY_GUIDE.md

---

## Example: Workshop 1 Reference

See `workshops/01-tokenization/` for the gold standard implementation:
- `tokenizer.py` - Three strategies with alien analogy
- `app.py` - Interactive demo with comparison tab
- `test_tokenizer.py` - 19 tests across 4 categories
- Complete slides, README, cheatsheet, Q&A
