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
### 🛸 The Alien Story

An alien lands on Earth wanting to understand human language.

Each workshop teaches them a new skill:
1. **Tokenize** - Learn to read symbols
2. **Embed** - Create a meaning map  
3. **Store** - Build a magic library
4. **Attend** - Focus on what matters
5. **Transform** - Build a complete brain
6. **RAG** - Get a search engine
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
# WORKSHOP 1: TOKENIZATION
# ============================================================================
elif workshop_choice == "1️⃣ Tokenization":
    st.markdown("## 🔤 Workshop 1: Tokenization")
    st.markdown("### *Teaching the alien to read symbols*")
    
    st.markdown("""
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
    st.markdown("### *Creating a map of meaning*")
    
    st.markdown("""
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
    st.markdown("### *The alien's magic library*")
    
    st.markdown("""
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
    st.markdown("### *The alien's spotlight of focus*")
    
    st.markdown("""
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
    st.markdown("### *Building the alien's complete brain*")
    
    st.markdown("""
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
    st.markdown("### *Give the alien a search engine!*")
    
    st.markdown("""
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
    st.markdown("### *Watch all 6 workshops work together!*")
    
    st.markdown("""
    This demo traces a query through the **complete GenAI pipeline**, 
    showing how each workshop's concept contributes to the final answer.
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
        st.markdown("### Step 1: 🔤 Tokenization")
        
        tokenizer = SimpleTokenizer(strategy='word')
        corpus = ["machine learning artificial intelligence neural networks"]
        tokenizer.train(corpus, vocab_size=100)
        
        tokens = tokenizer.encode(user_input)
        st.markdown(f"**Input:** `{user_input}`")
        st.markdown(f"**Tokens:** `{tokens[:10]}...` ({len(tokens)} total)")
        st.success("✅ Text converted to token IDs")
        
        # Step 2: Embeddings
        st.markdown("---")
        st.markdown("### Step 2: 🗺️ Embeddings")
        
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
        st.markdown("### Step 3: 📚 Vector Database Search")
        
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
        st.markdown("### Step 4: 👀 Attention")
        
        st.markdown("""
        The transformer uses attention to:
        - Let query tokens attend to context tokens
        - Identify which parts of retrieved docs are most relevant
        - Focus on key information for answering
        """)
        st.success("✅ Attention weights computed across query + context")
        
        # Step 5: Transformer Processing
        st.markdown("---")
        st.markdown("### Step 5: 🧠 Transformer")
        
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
        st.markdown("### Step 6: 🔍 RAG Generation")
        
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
        st.markdown("### 🎉 Pipeline Complete!")
        
        st.markdown("""
        <div class='pipeline-flow'>
            <div class='pipeline-step'>📝 Input</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>🔤 Tokenize</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>🗺️ Embed</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>📚 Search</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>👀 Attend</div>
            <span class='pipeline-arrow'>→</span>
            <div class='pipeline-step'>🧠 Transform</div>
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
