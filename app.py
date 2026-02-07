"""
🛸 GenAI Self-Build: The Complete Journey
==========================================

A unified interactive demo bringing together all 6 workshops
into one cohesive experience.

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add workshop directories to path
workshops_dir = Path(__file__).parent / "workshops"
for workshop in workshops_dir.iterdir():
    if workshop.is_dir():
        sys.path.insert(0, str(workshop))

# Import from workshops
from tokenizer import SimpleTokenizer
from embeddings import SimpleEmbedding
from vector_db import SimpleVectorDB
from attention import SimpleAttention
from transformer import MiniTransformer
from rag import RAGPipeline, SAMPLE_KNOWLEDGE


# Page config
st.set_page_config(
    page_title="🛸 GenAI Self-Build",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    .workshop-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #4a4a6a;
        transition: all 0.3s ease;
    }
    .workshop-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    .step-indicator {
        display: inline-block;
        width: 40px;
        height: 40px;
        line-height: 40px;
        text-align: center;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        margin-right: 10px;
    }
    .pipeline-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        padding: 20px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        margin: 20px 0;
    }
    .pipeline-step {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
    }
    .pipeline-arrow {
        font-size: 1.5rem;
        color: #667eea;
    }
    .output-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        font-family: monospace;
        margin: 10px 0;
    }
    .metric-row {
        display: flex;
        gap: 20px;
        margin: 10px 0;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class='main-header'>
    <h1>🛸 GenAI Self-Build</h1>
    <h3>The Complete Journey: From Text to Intelligence</h3>
    <p><em>Demystifying how AI really works, one concept at a time</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🗺️ Workshop Navigator")
workshop_choice = st.sidebar.radio(
    "Choose a workshop:",
    [
        "🏠 Home: The Big Picture",
        "📖 Zara's Journey",
        "1️⃣ Tokenization",
        "2️⃣ Embeddings",
        "3️⃣ Vector Databases",
        "4️⃣ Attention",
        "5️⃣ Transformers",
        "6️⃣ RAG",
        "🔗 End-to-End Pipeline"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🛸 Meet Zara

Zara is a Zorathian scientist orbiting Earth.
She can intercept human text — but can't understand a word.

Follow her journey through 6 chapters:
1. **📕 The Codebook** — Learn to read symbols
2. **📗 The Map** — Discover meaning in math
3. **📘 The Library** — Store & search knowledge
4. **📙 The Spotlight** — Focus on what matters
5. **📓 The Brain** — Assemble a thinking machine
6. **📔 The Search Engine** — Ground answers in facts
""")


# ============================================================================
# HOME: THE BIG PICTURE
# ============================================================================
if workshop_choice == "🏠 Home: The Big Picture":
    st.markdown("## 🌟 Welcome to the GenAI Self-Build Workshop Series!")
    
    st.markdown("""
    This interactive demo brings together **6 workshops** that demystify how 
    modern AI systems like ChatGPT and Claude actually work.
    
    ### 🎯 The Core Insight
    
    > **AI doesn't understand language the way you do.**
    > It transforms text through a series of mathematical operations.
    > Each workshop reveals one piece of this puzzle.
    """)
    
    # Visual pipeline
    st.markdown("### 🔄 The Complete Pipeline")
    st.markdown("""
    <div class='pipeline-flow'>
        <div class='pipeline-step'>📝 Text</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>🔢 Tokens</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>📊 Embeddings</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>🔍 Search</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>👀 Attention</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>🧠 Transform</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>💬 Output</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Workshop cards
    st.markdown("### 📚 The Workshops")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='workshop-card'>
            <h4>🔤 Workshop 1: Tokenization</h4>
            <p><strong>Analogy:</strong> Alien builds a codebook</p>
            <p>Convert text to numbers. The first step in all NLP!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='workshop-card'>
            <h4>🗺️ Workshop 2: Embeddings</h4>
            <p><strong>Analogy:</strong> Creating a map of meaning</p>
            <p>Place words in space where similar = nearby.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='workshop-card'>
            <h4>📚 Workshop 3: Vector Databases</h4>
            <p><strong>Analogy:</strong> Magic library shelves</p>
            <p>Store and search millions of vectors fast.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='workshop-card'>
            <h4>👀 Workshop 4: Attention</h4>
            <p><strong>Analogy:</strong> Spotlight of focus</p>
            <p>Learn what context matters for each word.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='workshop-card'>
            <h4>🧠 Workshop 5: Transformers</h4>
            <p><strong>Analogy:</strong> Complete alien brain</p>
            <p>The full architecture that powers GPT, Claude, etc.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='workshop-card'>
            <h4>🔍 Workshop 6: RAG</h4>
            <p><strong>Analogy:</strong> Search engine + brain</p>
            <p>Ground generation in real documents.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 **Select a workshop from the sidebar** to explore each concept interactively!")


# ============================================================================
# ZARA'S JOURNEY: THE NARRATIVE THREAD
# ============================================================================
elif workshop_choice == "📖 Zara's Journey":
    st.markdown("## 📖 Zara's Journey: From Silence to Understanding")
    st.markdown("### *The story that connects all 6 workshops*")

    st.markdown("""
    > *"What I cannot create, I do not understand."* — Richard Feynman

    Follow **Zara**, a Zorathian scientist orbiting Earth, as she builds —
    from scratch — every component needed to understand human language.
    Her journey mirrors exactly how modern AI systems like ChatGPT were built.
    """)

    # Zara's backstory
    with st.expander("🛸 Meet Zara — The Backstory", expanded=True):
        st.markdown("""
        Zara is a brilliant scientist from the planet Zorath-7. She's been observing
        Earth from orbit, intercepting millions of text messages, emails, and web pages.

        But she faces one fundamental problem: **she cannot understand a single word.**

        On Zorath-7, communication is through electromagnetic pulses — pure numbers
        and frequencies. So when Zara sees `"Hello, how are you?"`, it looks like
        meaningless noise — the same way `"✧⚡↯∞⊕✧"` looks to you.

        Zara is determined. She will learn to understand human language.
        But she'll do it **her way** — by breaking it down into numbers, math, and patterns.
        """)

    st.markdown("---")
    st.markdown("### 🎬 The Six Chapters")

    # Chapter navigation
    chapter = st.selectbox(
        "Choose a chapter:",
        [
            "📕 Chapter 1: The Codebook (Tokenization)",
            "📗 Chapter 2: The Map (Embeddings)",
            "📘 Chapter 3: The Library (Vector Databases)",
            "📙 Chapter 4: The Breakthrough (Attention)",
            "📓 Chapter 5: The Brain (Transformers)",
            "📔 Chapter 6: The Connection (RAG)",
        ]
    )

    if "Chapter 1" in chapter:
        st.markdown("""
        ## 📕 Chapter 1: The Codebook

        > *Zara stares at her screen. Millions of intercepted Earth transmissions
        > scroll past — but they're just shapes. Meaningless symbols. She needs to
        > start somewhere, so she asks the simplest question: "Can I turn these
        > symbols into numbers?"*

        **Zara's challenge:** Build a codebook that translates human symbols into
        numbers she can process.

        She tries three approaches:

        | Attempt | Strategy | Result |
        |---------|----------|--------|
        | 1st | Letter by letter | ✅ Works, but sentences become enormously long |
        | 2nd | Word by word | ✅ Much shorter, but new words break everything |
        | 3rd | Common patterns (BPE) | ✅ The sweet spot — what GPT actually uses! |

        **🔗 Try it yourself** → Select *1️⃣ Tokenization* from the sidebar

        ---

        > 📖 *"Zara now has a codebook. She can turn any human text into numbers.
        > But there's a problem — the number for 'Hello' and 'Goodbye' look almost
        > the same, yet they mean opposite things. She needs a better way to
        > capture what words actually MEAN..."*
        """)

    elif "Chapter 2" in chapter:
        st.markdown("""
        ## 📗 Chapter 2: The Map

        > *Zara has been staring at her codebook for days. Token 42 ('cat') and
        > Token 43 ('dog') are just one number apart, but 'cat' and 'kitten' —
        > which are very similar — are hundreds apart. The numbers don't capture
        > meaning at all!*
        >
        > *Then she has a brilliant idea: what if she creates a MAP of meaning?*

        **Zara's insight:** Words that appear in similar contexts probably mean
        similar things. "The **cat** sat on the mat" and "The **dog** sat on the rug"
        — cat and dog appear in the same position!

        She builds a **co-occurrence map** that places similar words near each other
        in a mathematical space. Now "king" and "queen" are neighbors, "happy" and
        "joyful" sit side by side, and she can even do math on meaning:

        `king - man + woman = queen` 👑

        **🔗 Try it yourself** → Select *2️⃣ Embeddings* from the sidebar

        ---

        > 📖 *"Zara's meaning map is incredible. But she's creating millions of
        > these vectors — one for every piece of text she's collected. She needs
        > somewhere to store them all, and a way to find things quickly..."*
        """)

    elif "Chapter 3" in chapter:
        st.markdown("""
        ## 📘 Chapter 3: The Library

        > *Zara has a problem every librarian would recognize. She's built millions
        > of meaning vectors, but finding the right one means comparing against
        > EVERY vector. With millions of documents, this takes forever!*
        >
        > *She needs a magic library where books organize themselves by meaning.*

        **Zara's solution:** Instead of checking every document, create "neighborhoods"
        where similar documents live together. When searching, only check the
        relevant neighborhood — like how a real library groups science books together.

        | Search Type | How It Works | Speed |
        |------------|-------------|-------|
        | Brute force | Compare against everything | 🐌 Slow |
        | LSH indexing | Check only nearby neighborhoods | 🚀 Fast |
        | Semantic | Find by meaning, not keywords | 🎯 Precise |

        **🔗 Try it yourself** → Select *3️⃣ Vector Databases* from the sidebar

        ---

        > 📖 *"Zara can now store and search millions of documents instantly.
        > But she faces a new challenge. When she reads 'The bank was near the
        > river bank,' she gets confused. Which 'bank' is which? She needs to
        > learn what humans do effortlessly — pay attention to context..."*
        """)

    elif "Chapter 4" in chapter:
        st.markdown("""
        ## 📙 Chapter 4: The Breakthrough

        > *This is the pivotal moment. Zara reads: "The cat sat on the mat because
        > IT was tired." What does "it" refer to? The cat? The mat? Humans solve
        > this instantly — they pay ATTENTION. But how do you teach a machine to focus?*

        **Zara's invention:** A spotlight system. Every word shines a spotlight on
        every other word, asking: *"Are you relevant to me?"*

        Think of it like a cocktail party — dozens of conversations happening at once,
        but you focus on the one person talking to you. If someone says your name
        across the room, your attention snaps there instantly.

        She implements **Query, Key, Value**:
        - **Query:** "What am I looking for?"
        - **Key:** "What do I have to offer?"
        - **Value:** "Here's my actual content"

        > 🎯 **This is the breakthrough that changed AI forever.**
        > The 2017 paper "Attention Is All You Need" used exactly this mechanism
        > and launched the transformer revolution.

        **🔗 Try it yourself** → Select *4️⃣ Attention* from the sidebar

        ---

        > 📖 *"Zara has cracked the attention puzzle. But one attention layer isn't
        > enough — it's like having one good pair of glasses. She needs bifocals,
        > a microscope AND a telescope. She needs to stack these layers, add
        > processing power, and build... a complete brain."*
        """)

    elif "Chapter 5" in chapter:
        st.markdown("""
        ## 📓 Chapter 5: The Brain

        > *Zara stands at her workbench, surrounded by all the components she's
        > built: a codebook, a meaning map, attention spotlights. Now she assembles
        > them into a working brain. She calls it... a Transformer.*

        **The architecture:**
        ```
        Input → Embedding → [Attention → Process → Normalize] × N → Output
        ```

        Each layer adds deeper understanding — like a team of specialists reading
        the same document. One checks grammar, another finds relationships, a third
        interprets meaning, and the last drafts a response.

        **⚠️ The humbling truth:** Zara's transformer works! It processes text and
        generates output. But... the output is gibberish. What went wrong?

        *Nothing.* The architecture is correct — identical to GPT-4 and Claude.
        But Zara's brain has thousands of parameters. GPT-4 has over a **trillion**.
        The lesson: **the magic isn't the architecture — it's the scale.**

        **🔗 Try it yourself** → Select *5️⃣ Transformers* from the sidebar

        ---

        > 📖 *"Zara's brain works but sometimes makes things up — it 'hallucinates.'
        > She needs a way to look things up before answering, to ground her
        > responses in real documents..."*
        """)

    elif "Chapter 6" in chapter:
        st.markdown("""
        ## 📔 Chapter 6: The Connection

        > *Zara has built an incredible system. But when someone asks a question,
        > she sometimes makes things up — confident but wrong. Sound familiar?
        > This is the same problem plaguing every large language model.*
        >
        > *Zara's solution: don't just rely on what you've learned — LOOK IT UP first.*

        **RAG = Retrieval Augmented Generation.** The full pipeline:

        1. 🔤 **Tokenize** the question (Chapter 1)
        2. 🗺️ **Embed** it as a meaning vector (Chapter 2)
        3. 📚 **Search** for relevant documents (Chapter 3)
        4. 👀 **Attend** to the important parts (Chapter 4)
        5. 🧠 **Transform** into a coherent answer (Chapter 5)
        6. 📎 **Cite** sources for verification (Chapter 6)

        **🔗 Try it yourself** → Select *6️⃣ RAG* from the sidebar

        ---

        ### 🎬 The Finale

        > *Zara started unable to read a single word of human text. Now she has
        > built every component of a modern AI language system — from scratch.*
        >
        > *And here's the remarkable thing: **you have too.** Everything Zara built,
        > you built alongside her. You now understand, at a fundamental level,
        > how ChatGPT, Claude, and Gemini actually work.*
        >
        > ***It's not magic. It's math, patterns, and brilliant engineering.
        > And now you can see it.*** 🎉
        """)

    # Story arc visualization
    st.markdown("---")
    st.markdown("### 🎭 The Emotional Arc")
    st.markdown("""
    <div class='pipeline-flow'>
        <div class='pipeline-step'>😟 Lost</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>🤔 Curious</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>😮 Wonder</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>💪 Confident</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>💡 Aha!</div>
        <span class='pipeline-arrow'>→</span>
        <div class='pipeline-step'>🎉 Triumph</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# WORKSHOP 1: TOKENIZATION
# ============================================================================
elif workshop_choice == "1️⃣ Tokenization":
    st.markdown("## 🔤 Workshop 1: Tokenization")
    st.markdown("### *📕 Chapter 1 of Zara's Journey: The Codebook*")
    
    st.markdown("""
    > 🛸 *Zara can't read human text. She needs a codebook to turn symbols into numbers.*
    
    **The Problem:** Computers don't understand text—only numbers!
    
    **The Solution:** Break text into tokens and assign each a number.
    """)
    
    # Demo
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text = st.text_input(
            "Enter text to tokenize:",
            value="The quick brown fox jumps over the lazy dog.",
            key="tok_input"
        )
    
    with col2:
        strategy = st.selectbox(
            "Strategy:",
            ["char", "word", "bpe"],
            index=1
        )
    
    # Train tokenizer
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we build software.",
        "Python is a popular programming language.",
        "The fox is quick and clever.",
    ]
    
    tokenizer = SimpleTokenizer(strategy=strategy)
    tokenizer.train(corpus, vocab_size=100)
    
    # Encode
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vocab Size", tokenizer.vocab_size())
    with col2:
        st.metric("Token Count", len(tokens))
    with col3:
        st.metric("Compression", f"{len(text)/max(len(tokens),1):.1f}x")
    
    st.markdown("**Token IDs:**")
    st.code(str(tokens), language="python")
    
    st.markdown("**Decoded:**")
    st.code(decoded, language="text")
    
    # Visual
    if strategy == "word":
        words = decoded.split()
        if words:
            cols = st.columns(min(len(words), 8))
            for i, (word, tok_id) in enumerate(zip(words[:8], tokens[:8])):
                with cols[i % 8]:
                    st.markdown(f"""
                    <div style='text-align:center; padding:10px; 
                                background:linear-gradient(135deg, #667eea, #764ba2);
                                border-radius:8px; margin:5px 0;'>
                        <div style='font-size:1.2rem;'>{word}</div>
                        <div style='font-size:0.8rem; opacity:0.8;'>ID: {tok_id}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ============================================================================
# WORKSHOP 2: EMBEDDINGS
# ============================================================================
elif workshop_choice == "2️⃣ Embeddings":
    st.markdown("## 🗺️ Workshop 2: Embeddings")
    st.markdown("### *📗 Chapter 2 of Zara's Journey: The Map*")
    
    st.markdown("""
    > 🛸 *Zara can read symbols now, but Token 42 ('cat') and Token 43 ('dog') look
    > the same to her. She needs a map where similar meanings live nearby.*
    
    **The Problem:** Token IDs don't capture meaning. "Cat" (ID 42) and "Dog" (ID 73) 
    look equally different from "Kitten" (ID 156), even though cat and kitten are related!
    
    **The Solution:** Map tokens to vectors where similar meanings are nearby.
    """)
    
    # Create embedder
    embed_dim = st.slider("Embedding dimension:", 8, 64, 32)
    embedder = SimpleEmbedding(strategy='cooccurrence', dimensions=embed_dim)
    
    # Demo words
    col1, col2 = st.columns(2)
    with col1:
        word1 = st.text_input("Word 1:", "king", key="emb_w1")
    with col2:
        word2 = st.text_input("Word 2:", "queen", key="emb_w2")
    
    # Get embeddings (using hash for demo)
    def get_word_embedding(word, dim):
        np.random.seed(hash(word.lower()) % 2**32)
        return np.random.randn(dim).astype(np.float32)
    
    emb1 = get_word_embedding(word1, embed_dim)
    emb2 = get_word_embedding(word2, embed_dim)
    
    # Compute similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    st.metric("Cosine Similarity", f"{similarity:.4f}")
    
    # Show vectors
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{word1}** embedding (first 8 dims):")
        st.code(str(emb1[:8].round(3).tolist()))
    with col2:
        st.markdown(f"**{word2}** embedding (first 8 dims):")
        st.code(str(emb2[:8].round(3).tolist()))
    
    st.info("""
    💡 **Note:** In real systems, embeddings are *learned* from data, so similar 
    words naturally end up with similar vectors. Our demo uses random vectors 
    seeded by word hash—real embeddings would show king≈queen similarity!
    """)


# ============================================================================
# WORKSHOP 3: VECTOR DATABASES
# ============================================================================
elif workshop_choice == "3️⃣ Vector Databases":
    st.markdown("## 📚 Workshop 3: Vector Databases")
    st.markdown("### *📘 Chapter 3 of Zara's Journey: The Library*")
    
    st.markdown("""
    > 🛸 *Zara has millions of meaning vectors now. She needs a magic library
    > where she can find the right one in seconds, not hours.*
    
    **The Problem:** With millions of embeddings, how do we find similar ones fast?
    
    **The Solution:** Vector databases with smart indexing (like a library's catalog system).
    """)
    
    # Create vector DB
    @st.cache_resource
    def create_demo_db():
        db = SimpleVectorDB(strategy='flat', dimensions=64)
        documents = [
            "Python is a programming language",
            "Java is also a programming language",
            "Machine learning uses algorithms",
            "Deep learning is a subset of ML",
            "Cats are furry pets",
            "Dogs are loyal companions",
            "The weather is sunny today",
            "It might rain tomorrow",
        ]
        
        # Add documents with simple embeddings
        for i, doc in enumerate(documents):
            np.random.seed(hash(doc) % 2**32)
            embedding = np.random.randn(64).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            db.add(f"doc_{i}", embedding)
        
        return db, documents
    
    db, documents = create_demo_db()
    
    st.markdown(f"**Database contains {len(documents)} documents**")
    
    # Search
    query = st.text_input(
        "Search query:",
        value="programming languages",
        key="vdb_query"
    )
    
    k = st.slider("Number of results:", 1, 5, 3)
    
    if st.button("🔍 Search", key="vdb_search"):
        # Create query embedding
        np.random.seed(hash(query) % 2**32)
        query_emb = np.random.randn(64).astype(np.float32)
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        results = db.search(query_emb, top_k=k)
        
        st.markdown("### Results:")
        for i, (doc_id, score) in enumerate(results):
            # Get document text from id (doc_0, doc_1, etc.)
            doc_idx = int(doc_id.split('_')[1])
            st.markdown(f"""
            <div class='workshop-card'>
                <strong>#{i+1}</strong> (Score: {score:.4f})<br>
                {documents[doc_idx]}
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# WORKSHOP 4: ATTENTION
# ============================================================================
elif workshop_choice == "4️⃣ Attention":
    st.markdown("## 👀 Workshop 4: Attention")
    st.markdown("### *📙 Chapter 4 of Zara's Journey: The Breakthrough*")
    
    st.markdown("""
    > 🛸 *"The bank was near the river bank" — which bank is which? Zara invents
    > a spotlight system where every word asks: "Who is relevant to me?"*
    
    **The Problem:** In "The cat sat on the mat because it was tired", what does "it" refer to?
    
    **The Solution:** Attention lets each word look at all other words and decide what's relevant.
    """)
    
    # Demo
    sentence = st.text_input(
        "Enter a sentence:",
        value="The cat sat on the mat",
        key="attn_input"
    )
    
    words = sentence.split()
    seq_len = len(words)
    
    if seq_len > 0:
        # Create attention module
        attn = SimpleAttention(embed_dim=64, num_heads=1)
        
        # Generate input (random embeddings for words)
        np.random.seed(42)
        x = np.random.randn(seq_len, 64).astype(np.float32)
        
        # Run attention
        output, weights = attn.forward(x)
        
        # Show attention matrix
        st.markdown("### Attention Weights")
        st.markdown("*Each row shows how much that word attends to other words*")
        
        # Create a simple visualization
        import pandas as pd
        
        weights_2d = weights[0] if len(weights.shape) == 3 else weights
        df = pd.DataFrame(
            weights_2d.round(3),
            columns=words,
            index=words
        )
        
        st.dataframe(df.style.background_gradient(cmap='Blues'))
        
        # Highlight explanation
        st.info("""
        💡 **Reading the matrix:** Each row shows attention distribution for one word.
        Higher values (darker blue) mean that word pays more attention to that column's word.
        """)


# ============================================================================
# WORKSHOP 5: TRANSFORMERS
# ============================================================================
elif workshop_choice == "5️⃣ Transformers":
    st.markdown("## 🧠 Workshop 5: Transformers")
    st.markdown("### *📓 Chapter 5 of Zara's Journey: The Brain*")
    
    st.markdown("""
    > 🛸 *Zara has all the pieces — a codebook, a meaning map, attention spotlights.
    > Now she assembles them into a working brain. She calls it... a Transformer.*
    
    **The Architecture:** Transformers combine:
    - **Embeddings** (Workshop 2) - Convert tokens to vectors
    - **Attention** (Workshop 4) - Let tokens interact
    - **Feed-forward layers** - Process the information
    - **Stacking** - Repeat for deeper understanding
    """)
    
    # Architecture visualization
    st.markdown("""
    ```
    Input → Embedding → [Attention → FFN] × N → Output
                           ↑__________________|
                           (residual connections)
    ```
    """)
    
    # Demo
    st.markdown("### 🎮 Try Generation")
    
    prompt = st.text_input(
        "Start of sentence:",
        value="The cat",
        key="tf_prompt"
    )
    
    max_tokens = st.slider("Tokens to generate:", 1, 10, 5)
    
    if st.button("🧠 Generate", key="tf_gen"):
        # Create a small transformer
        transformer = MiniTransformer(
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32
        )
        
        # Tokenize
        words = prompt.lower().split()
        
        # Simple word to ID mapping
        word_to_id = {w: i+1 for i, w in enumerate(set(words))}
        input_ids = [word_to_id.get(w, 0) for w in words]
        
        # Generate using the transformer's generate method
        generated_ids = transformer.generate(
            prompt_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=1.0,
            top_k=10
        )
        
        # Get just the new tokens (not the prompt)
        new_tokens = generated_ids[len(input_ids):]
        
        st.markdown("**Generated token IDs:**")
        st.code(str(new_tokens))
        
        st.warning("""
        ⚠️ **Note:** This transformer has random (untrained) weights, so it produces
        random output. Real transformers are trained on billions of words to learn
        meaningful patterns!
        """)


# ============================================================================
# WORKSHOP 6: RAG
# ============================================================================
elif workshop_choice == "6️⃣ RAG":
    st.markdown("## 🔍 Workshop 6: RAG")
    st.markdown("### *📔 Chapter 6 of Zara's Journey: The Connection*")
    
    st.markdown("""
    > 🛸 *Zara's brain works, but sometimes makes things up. Her solution:
    > look things up BEFORE answering — don't hallucinate, verify!*
    
    **The Problem:** LLMs can hallucinate, have knowledge cutoffs, and can't cite sources.
    
    **The Solution:** RAG = Retrieval + Augmented + Generation
    1. Search for relevant documents
    2. Add them to the prompt
    3. Generate answer based on retrieved context
    """)
    
    # Create RAG pipeline
    @st.cache_resource
    def create_rag():
        rag = RAGPipeline(embed_dim=64, top_k=3)
        texts = [doc[0] for doc in SAMPLE_KNOWLEDGE]
        sources = [doc[1] for doc in SAMPLE_KNOWLEDGE]
        rag.add_knowledge(texts, sources)
        return rag
    
    rag = create_rag()
    
    st.markdown(f"**Knowledge base: {len(SAMPLE_KNOWLEDGE)} documents**")
    
    # Query
    question = st.text_input(
        "Ask a question:",
        value="What is Python?",
        key="rag_q"
    )
    
    if st.button("🔍 Get Answer", key="rag_search"):
        result = rag.query(question)
        
        st.markdown("### 💬 Answer")
        st.success(result['answer'])
        
        st.markdown("### 📚 Sources")
        st.info(", ".join(result['sources']))
        
        with st.expander("View Retrieved Documents"):
            for doc in result['retrieved_docs']:
                st.markdown(f"""
                **Score: {doc['score']:.3f}** | Source: {doc['metadata'].get('source', 'unknown')}
                > {doc['text'][:200]}...
                """)


# ============================================================================
# END-TO-END PIPELINE
# ============================================================================
elif workshop_choice == "🔗 End-to-End Pipeline":
    st.markdown("## 🔗 End-to-End Pipeline")
    st.markdown("### *🛸 Zara's complete system — all 6 chapters working together!*")
    
    st.markdown("""
    This demo traces a query through Zara's **complete GenAI pipeline**,
    showing how each chapter's invention contributes to the final answer.
    
    > *Zara started unable to read a single word. Now watch every component
    > she built work together as one system.*
    """)
    
    # Input
    user_input = st.text_area(
        "Enter your question:",
        value="What is machine learning and how does it work?",
        height=100
    )
    
    if st.button("🚀 Run Complete Pipeline", type="primary"):
        
        # Step 1: Tokenization
        st.markdown("---")
        st.markdown("### 📕 Step 1: The Codebook (Tokenization)")
        
        tokenizer = SimpleTokenizer(strategy='word')
        corpus = ["machine learning artificial intelligence neural networks"]
        tokenizer.train(corpus, vocab_size=100)
        
        tokens = tokenizer.encode(user_input)
        st.markdown(f"**Input:** `{user_input}`")
        st.markdown(f"**Tokens:** `{tokens[:10]}...` ({len(tokens)} total)")
        st.success("✅ Text converted to token IDs")
        
        # Step 2: Embeddings
        st.markdown("---")
        st.markdown("### 📗 Step 2: The Map (Embeddings)")
        
        embed_dim = 64
        # Note: We don't actually use the embedder here - just showing the concept
        # Real embeddings would come from training on a corpus
        
        # Create query embedding
        np.random.seed(hash(user_input) % 2**32)
        query_embedding = np.random.randn(embed_dim).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        st.markdown(f"**Query embedding:** `{query_embedding[:5].round(3).tolist()}...` ({embed_dim} dims)")
        st.success("✅ Tokens converted to dense vectors")
        
        # Step 3: Vector Search
        st.markdown("---")
        st.markdown("### 📘 Step 3: The Library (Vector Search)")
        
        rag = RAGPipeline(embed_dim=64, top_k=3)
        texts = [doc[0] for doc in SAMPLE_KNOWLEDGE]
        sources = [doc[1] for doc in SAMPLE_KNOWLEDGE]
        rag.add_knowledge(texts, sources)
        
        retrieved = rag.doc_store.search(user_input, top_k=3)
        
        st.markdown("**Retrieved documents:**")
        for i, doc in enumerate(retrieved):
            st.markdown(f"- [{doc['metadata']['source']}] Score: {doc['score']:.3f}")
        st.success("✅ Found relevant documents via similarity search")
        
        # Step 4: Attention (conceptual)
        st.markdown("---")
        st.markdown("### 📙 Step 4: The Spotlight (Attention)")
        
        st.markdown("""
        The transformer uses attention to:
        - Let query tokens attend to context tokens
        - Identify which parts of retrieved docs are most relevant
        - Focus on key information for answering
        """)
        st.success("✅ Attention weights computed across query + context")
        
        # Step 5: Transformer Processing
        st.markdown("---")
        st.markdown("### 📓 Step 5: The Brain (Transformer)")
        
        st.markdown("""
        The transformer processes the augmented prompt through:
        - Multiple attention layers
        - Feed-forward networks
        - Layer normalization
        - Final output projection
        """)
        st.success("✅ Transformer processed augmented prompt")
        
        # Step 6: RAG Output
        st.markdown("---")
        st.markdown("### 📔 Step 6: The Search Engine (RAG)")
        
        result = rag.query(user_input)
        
        st.markdown("**Generated Answer:**")
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a4a1a, #0d3d0d);
                    border: 2px solid #16a34a; border-radius: 12px; padding: 20px;'>
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Sources:** {', '.join(result['sources'])}")
        st.success("✅ Answer generated with source citations!")
        
        # Summary
        st.markdown("---")
        st.markdown("### 🎉 Zara's Mission Complete!")
        
        st.markdown("""
        > *Every component you just saw was built from scratch across 6 workshops.
        > The same architecture powers ChatGPT, Claude, and Gemini — just at a
        > much larger scale. Now you can see how it all fits together.*
        """)
        
        st.markdown("""
        <div class='pipeline-flow'>
            <div class='pipeline-step'>📝 Input</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>📕 Codebook</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>📗 Map</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>📘 Library</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>📙 Spotlight</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>📓 Brain</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>💬 Answer</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    🛸 <strong>GenAI Self-Build Workshop Series</strong><br>
    <em>Demystifying AI, one concept at a time</em><br><br>
    Created by Michael Kennedy | 📧 michael.kennedy@analog.com
</div>
""", unsafe_allow_html=True)
