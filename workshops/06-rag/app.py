"""
🔍 Workshop 6: RAG Explorer
============================

Interactive Streamlit app to explore Retrieval-Augmented Generation.

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from rag import (
    SimpleEmbedder, DocumentStore, PromptBuilder,
    SimpleGenerator, RAGPipeline, SAMPLE_KNOWLEDGE
)


# Page config
st.set_page_config(
    page_title="🔍 RAG Explorer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-size: 1.1rem;
    }
    .doc-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #4a4a6a;
    }
    .doc-source {
        color: #667eea;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .doc-text {
        color: #e0e0e0;
        margin-top: 8px;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        float: right;
    }
    .pipeline-step {
        background: rgba(100, 126, 234, 0.1);
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .answer-box {
        background: linear-gradient(135deg, #1a4a1a 0%, #0d3d0d 100%);
        border: 2px solid #16a34a;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    .sources-box {
        background: linear-gradient(135deg, #1a1a4a 0%, #0d0d3d 100%);
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 RAG Explorer")
st.markdown("### *Give the alien a search engine for their brain!*")

# Sidebar
with st.sidebar:
    st.markdown("## 🛸 The Search Engine Analogy")
    st.markdown("""
    Our alien's brain (transformer) has a problem:
    - Only knows what it learned in "school" (training)
    - Can't access new information
    - Might make things up!
    
    **Solution: Build a library!**
    
    1. 📝 User asks a question
    2. 🔍 Search the library for relevant books
    3. 📚 Pull out the most relevant pages
    4. 🧠 Read pages + question together
    5. 💬 Answer based on what was just read
    
    This is **RAG**:
    - **R**etrieval: Find docs
    - **A**ugmented: Add to prompt
    - **G**eneration: Produce answer
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    top_k = st.slider("Documents to retrieve:", 1, 5, 3)
    embed_dim = st.select_slider("Embedding dimension:", [32, 64, 128], 64)
    
    st.markdown("---")
    st.markdown("### 📊 Workshop 6 of 6")
    st.markdown("*The Grand Finale!*")


# Initialize RAG pipeline
@st.cache_resource
def load_rag(embed_dim, top_k):
    rag = RAGPipeline(embed_dim=embed_dim, top_k=top_k)
    texts = [doc[0] for doc in SAMPLE_KNOWLEDGE]
    sources = [doc[1] for doc in SAMPLE_KNOWLEDGE]
    rag.add_knowledge(texts, sources)
    return rag


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Ask Questions",
    "🔬 Pipeline Walkthrough",
    "📚 Knowledge Base",
    "🏗️ How It Works"
])


# Tab 1: Ask Questions
with tab1:
    st.markdown("## 💬 Ask the RAG System")
    st.markdown("""
    Ask any question about the topics in our knowledge base. 
    The system will retrieve relevant documents and answer based on them.
    """)
    
    rag = load_rag(embed_dim, top_k)
    
    # Example questions
    st.markdown("**Try these example questions:**")
    example_cols = st.columns(3)
    example_questions = [
        "What is Python?",
        "How do transformers work?",
        "What is photosynthesis?",
        "When was World War II?",
        "How tall is Mount Everest?",
        "What is RAG?"
    ]
    
    selected_example = None
    for i, q in enumerate(example_questions):
        with example_cols[i % 3]:
            if st.button(q, key=f"example_{i}"):
                selected_example = q
    
    # Question input
    question = st.text_input(
        "Or type your own question:",
        value=selected_example if selected_example else "",
        placeholder="Ask me anything about the knowledge base..."
    )
    
    if st.button("🔍 Get Answer", type="primary") or selected_example:
        if question:
            with st.spinner("Retrieving and generating..."):
                result = rag.query(question)
            
            # Show answer
            st.markdown("### 💬 Answer")
            st.markdown(f"""
            <div class='answer-box'>
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources
            st.markdown("### 📚 Sources Used")
            st.markdown(f"""
            <div class='sources-box'>
                <strong>Retrieved from:</strong> {', '.join(result['sources'])}
            </div>
            """, unsafe_allow_html=True)
            
            # Show retrieved documents
            with st.expander("📄 View Retrieved Documents", expanded=False):
                for i, doc in enumerate(result['retrieved_docs']):
                    st.markdown(f"""
                    <div class='doc-card'>
                        <span class='score-badge'>Score: {doc['score']:.3f}</span>
                        <div class='doc-source'>{doc['metadata'].get('source', 'Unknown')}</div>
                        <div class='doc-text'>{doc['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show prompt (for learning purposes)
            with st.expander("📝 View Generated Prompt", expanded=False):
                st.code(result['prompt'], language="text")
        else:
            st.warning("Please enter a question!")


# Tab 2: Pipeline Walkthrough
with tab2:
    st.markdown("## 🔬 RAG Pipeline Step-by-Step")
    st.markdown("""
    Watch each step of the RAG process in detail.
    """)
    
    walkthrough_q = st.text_input(
        "Enter a question to trace through the pipeline:",
        value="What is machine learning?",
        key="walkthrough_q"
    )
    
    if st.button("🔬 Trace Pipeline", type="primary"):
        rag = load_rag(embed_dim, top_k)
        
        # Step 1: Embed Query
        st.markdown("### Step 1: Embed the Question")
        st.markdown(f"""
        <div class='pipeline-step'>
            <strong>Input:</strong> "{walkthrough_q}"<br>
            <strong>Output:</strong> Vector of {embed_dim} dimensions
        </div>
        """, unsafe_allow_html=True)
        
        query_emb = rag.embedder.embed(walkthrough_q)
        st.markdown(f"**Query embedding (first 8 dims):** `{query_emb[:8].round(3).tolist()}`")
        
        # Step 2: Search
        st.markdown("### Step 2: Search Vector Database")
        retrieved = rag.doc_store.search(walkthrough_q, top_k=top_k)
        
        st.markdown(f"""
        <div class='pipeline-step'>
            <strong>Query:</strong> Semantic search<br>
            <strong>Result:</strong> Top {len(retrieved)} most similar documents
        </div>
        """, unsafe_allow_html=True)
        
        for i, doc in enumerate(retrieved):
            st.markdown(f"""
            **[{i+1}]** Score: `{doc['score']:.3f}` | Source: `{doc['metadata'].get('source', 'unknown')}`
            > {doc['text'][:150]}...
            """)
        
        # Step 3: Build Prompt
        st.markdown("### Step 3: Build Augmented Prompt")
        prompt, sources = rag.prompt_builder.build_prompt_with_sources(walkthrough_q, retrieved)
        
        st.markdown(f"""
        <div class='pipeline-step'>
            <strong>Combine:</strong> Context + Question + Instructions<br>
            <strong>Result:</strong> Prompt of {len(prompt)} characters
        </div>
        """, unsafe_allow_html=True)
        
        st.code(prompt[:500] + "..." if len(prompt) > 500 else prompt, language="text")
        
        # Step 4: Generate
        st.markdown("### Step 4: Generate Answer")
        answer = rag.generator.generate(prompt)
        
        st.markdown(f"""
        <div class='pipeline-step'>
            <strong>Input:</strong> Augmented prompt<br>
            <strong>Output:</strong> Answer grounded in retrieved context
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"**Answer:** {answer}")
        st.info(f"**Sources:** {', '.join(sources)}")


# Tab 3: Knowledge Base
with tab3:
    st.markdown("## 📚 Knowledge Base Explorer")
    st.markdown("""
    Browse all documents in our knowledge base. In a real system, 
    this could be millions of documents!
    """)
    
    rag = load_rag(embed_dim, top_k)
    
    # Show statistics
    st.markdown("### 📊 Statistics")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(rag.doc_store.documents)}</div>
            <div class='metric-label'>Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{embed_dim}</div>
            <div class='metric-label'>Embedding Dim</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[2]:
        categories = set(d['metadata']['source'].split('/')[0] for d in rag.doc_store.documents)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(categories)}</div>
            <div class='metric-label'>Categories</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[3]:
        total_chars = sum(len(d['text']) for d in rag.doc_store.documents)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total_chars // 1000}K</div>
            <div class='metric-label'>Total Characters</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filter by category
    all_categories = sorted(set(d['metadata']['source'].split('/')[0] for d in rag.doc_store.documents))
    selected_cat = st.selectbox("Filter by category:", ["All"] + all_categories)
    
    # Show documents
    st.markdown("### 📄 Documents")
    for doc in rag.doc_store.documents:
        source = doc['metadata']['source']
        category = source.split('/')[0]
        
        if selected_cat != "All" and category != selected_cat:
            continue
        
        st.markdown(f"""
        <div class='doc-card'>
            <div class='doc-source'>📁 {source}</div>
            <div class='doc-text'>{doc['text']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add custom document
    st.markdown("---")
    st.markdown("### ➕ Add Custom Document")
    
    new_text = st.text_area("Document text:", placeholder="Enter your document content...")
    new_source = st.text_input("Source name:", placeholder="e.g., custom/my_doc")
    
    if st.button("Add Document"):
        if new_text and new_source:
            rag.add_knowledge([new_text], [new_source])
            st.success(f"Added document: {new_source}")
            st.rerun()
        else:
            st.warning("Please enter both text and source name.")


# Tab 4: How It Works
with tab4:
    st.markdown("## 🏗️ How RAG Works")
    
    # Visual pipeline
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #667eea, #764ba2); 
                    border-radius: 10px; color: white;'>
            ❓ Question
        </div>
        <span style='font-size: 2rem; margin: 0 10px;'>→</span>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #f093fb, #f5576c); 
                    border-radius: 10px; color: white;'>
            🔢 Embed
        </div>
        <span style='font-size: 2rem; margin: 0 10px;'>→</span>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #4facfe, #00f2fe); 
                    border-radius: 10px; color: white;'>
            🔍 Retrieve
        </div>
        <span style='font-size: 2rem; margin: 0 10px;'>→</span>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #fa709a, #fee140); 
                    border-radius: 10px; color: white;'>
            📝 Augment
        </div>
        <span style='font-size: 2rem; margin: 0 10px;'>→</span>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #43e97b, #38f9d7); 
                    border-radius: 10px; color: white;'>
            💬 Generate
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Why RAG?
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ❌ Problems with LLMs Alone
        
        1. **Hallucination**
           - Models can make things up
           - No way to verify claims
           - Confidently wrong
        
        2. **Staleness**
           - Training data has a cutoff
           - Can't know about recent events
           - Knowledge becomes outdated
        
        3. **Opacity**
           - No sources for claims
           - Hard to fact-check
           - Black box answers
        """)
    
    with col2:
        st.markdown("""
        ### ✅ How RAG Solves These
        
        1. **Grounded in Documents**
           - Answers based on retrieved text
           - Can verify against sources
           - Less likely to fabricate
        
        2. **Always Current**
           - Just update the knowledge base
           - No retraining needed
           - Real-time information
        
        3. **Transparent**
           - Sources cited with answers
           - Users can verify claims
           - Builds trust
        """)
    
    st.markdown("---")
    
    # Workshop connections
    st.markdown("""
    ### 🔗 How This Workshop Connects to the Series
    
    RAG brings together **everything we've learned**:
    
    | Workshop | Component | Role in RAG |
    |----------|-----------|-------------|
    | 1. Tokenization | Text → Tokens | Process input/output text |
    | 2. Embeddings | Tokens → Vectors | Encode queries and documents |
    | 3. Vector DB | Store & Search | Retrieve relevant documents |
    | 4. Attention | Focus on relevant parts | (Used inside transformer) |
    | 5. Transformer | Process sequences | Generate the final answer |
    | **6. RAG** | **Complete Pipeline** | **Put it all together!** |
    """)
    
    st.markdown("---")
    
    # Real-world examples
    st.markdown("""
    ### 🌍 RAG in the Real World
    
    - **Perplexity AI**: Search engine with LLM answers + sources
    - **ChatGPT Plugins**: Access real-time data via retrieval
    - **Enterprise AI**: Query internal documents securely
    - **Code Assistants**: Reference documentation and codebases
    - **Customer Support**: Answer questions from knowledge bases
    """)
    
    st.info("""
    🎉 **Congratulations!** You've completed the GenAI Self-Build Workshop Series!
    
    You now understand the core components that power modern AI systems:
    - How text becomes numbers (tokenization)
    - How meaning becomes vectors (embeddings)
    - How we find similar things (vector search)
    - How models focus attention
    - How transformers process language
    - How RAG grounds generation in knowledge
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    🔍 Workshop 6: RAG | Part of the GenAI Self-Build Series (6 of 6) | 🎉 The Grand Finale!
</div>
""", unsafe_allow_html=True)
