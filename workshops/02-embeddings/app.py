"""
🎯 Workshop 2: Interactive Embeddings Demo
==========================================

A Streamlit app to explore word embeddings interactively.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from embeddings import SimpleEmbedding

# Page config
st.set_page_config(
    page_title="🗺️ Embeddings Explorer",
    page_icon="🗺️",
    layout="wide"
)

# Custom CSS for visualizations
st.markdown("""
<style>
    .vector-box {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
        min-width: 50px;
        text-align: center;
    }
    .strategy-random { background-color: #6b7280; color: white; }
    .strategy-cooccurrence { background-color: #8b5cf6; color: white; }
    .strategy-prediction { background-color: #10b981; color: white; }
    .sim-bar {
        height: 20px;
        border-radius: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .alien-box {
        background-color: #1e1e2e;
        border: 2px solid #7c3aed;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .flow-step {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d1f3d 100%);
        border: 1px solid #7c3aed;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
    .flow-arrow {
        font-size: 24px;
        color: #7c3aed;
    }
    .info-card {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    .vector-positive {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    .vector-negative {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    .vector-neutral {
        background: linear-gradient(135deg, #6b7280, #4b5563);
        color: white;
    }
    .word-label {
        font-size: 12px;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 3px;
        background-color: rgba(0,0,0,0.7);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🗺️ Embeddings Explorer")
st.markdown("### *How does an alien map the meaning of words?*")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["🔬 Interactive Demo", "📊 How It Works", "🆚 Compare Strategies"])

# Map analogy in sidebar
with st.sidebar:
    st.markdown("## 🗺️ The Map Analogy")
    st.markdown("""
    Our alien friend from Workshop 1 has learned to read symbols.
    Now they need to understand what words **MEAN**.
    
    The alien creates a **MAP** where:
    - Similar words are **CLOSE** together
    - "King" lives near "queen", "prince"
    - "Cat" lives near "dog", "pet"
    
    The magical discovery: **meaning follows DIRECTIONS!**
    - King → Queen is the same as Man → Woman
    - So: king - man + woman = queen! 👑
    
    **This is exactly what embeddings do!**
    """)
    
    st.divider()
    
    st.markdown("## ⚙️ Settings")
    
    # Strategy selection
    strategy = st.radio(
        "Embedding Strategy",
        ["random", "cooccurrence", "prediction"],
        format_func=lambda x: {
            "random": "🎲 Random (baseline)",
            "cooccurrence": "🔗 Co-occurrence (context)",
            "prediction": "🎯 Prediction (neural)"
        }[x],
        help="Switch strategies to see different embedding approaches"
    )
    
    # Dimension slider
    dimensions = st.slider(
        "Vector Dimensions",
        min_value=10,
        max_value=100,
        value=100,
        step=10,
        help="More dimensions = more expressive, but harder to visualize"
    )
    
    # Strategy info
    strategy_info = {
        "random": ("🎲 Random", "Throws darts at the map - no real meaning!", "#6b7280"),
        "cooccurrence": ("🔗 Co-occurrence", "Words in similar contexts get similar vectors", "#8b5cf6"),
        "prediction": ("🎯 Prediction", "Learns by predicting neighboring words", "#10b981")
    }
    name, desc, color = strategy_info[strategy]
    st.markdown(f"""
    <div class="info-card" style="border-left-color: {color};">
        <strong>{name}</strong><br/>
        {desc}
    </div>
    """, unsafe_allow_html=True)
    
    # Clear cache button
    st.divider()
    if st.button("🔄 Retrain Embeddings", use_container_width=True, help="Force retrain with current settings"):
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    st.markdown("### 🎯 Workshop 2 of 6")
    st.markdown("*GenAI Self-Build Series*")

# Default corpus with strong semantic relationships - repeated patterns for better learning
DEFAULT_CORPUS = """The king and queen ruled the kingdom wisely.
The king and queen lived in the royal castle.
The king and queen attended the royal ceremony.
The prince and princess are the children of the king and queen.
The king wore a golden crown on his head.
The queen wore a silver crown on her head.
The king sat on the royal throne.
The queen sat beside the king on her throne.
The royal king ruled the kingdom.
The royal queen ruled beside the king.
The kingdom loved their king and queen.
The prince will become king and the princess may become queen.
The cat and dog are popular pets.
The cat and dog played in the yard.
The cat and dog are furry animals.
My pet cat sleeps on the soft bed.
My pet dog runs in the green park.
The cat chased the mouse while the dog barked.
A cat is a pet and a dog is a pet.
The puppy is a young dog and the kitten is a young cat.
Dogs and cats are common household pets.
The fluffy cat and the friendly dog.
Machine learning transforms data into insights.
Neural networks learn patterns from data.
Deep learning uses neural networks.
The model learns from training data.
Artificial intelligence powers many applications.
Data science combines statistics and programming.
Machine learning and deep learning use data.
Neural networks are used in machine learning.
The man and woman walked together.
The man and woman talked together.
The boy and girl played in the park.
The boy and girl are children.
He is a man and she is a woman.
A king is a man and a queen is a woman.
The king is a royal man.
The queen is a royal woman.
Man and woman are adults.
The man worked while the woman worked.
The king rules and the queen rules.
A cat purrs and a dog barks.
The man is strong and the woman is strong.
Happy people smile and laugh with joy.
Sad people feel unhappy and may cry.
Good things make us happy and feel positive.
Bad things make us sad and feel negative.
Joy and happiness are positive emotions.
Sadness and unhappiness are negative emotions."""

# Initialize embeddings (cached)
@st.cache_resource
def get_embedder(strategy, corpus_hash, dimensions):
    """Create and train an embedder with caching."""
    embedder = SimpleEmbedding(strategy=strategy, dimensions=dimensions)
    corpus = [line.strip() for line in corpus_hash.split('\n') if line.strip()]
    # Use more epochs for prediction strategy for better results
    epochs = 30 if strategy == 'prediction' else 10
    embedder.train(corpus, window_size=3, epochs=epochs)
    return embedder

# ============================================================================
# TAB 1: Interactive Demo
# ============================================================================
with tab1:
    # Data flow visualization at top
    st.markdown("## 🔄 The Embedding Pipeline")
    
    flow_cols = st.columns([2, 1, 2, 1, 2, 1, 2])
    
    with flow_cols[0]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">📝</div>
            <div><strong>Text</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Training corpus</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[1]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[2]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">🔢</div>
            <div><strong>Tokens</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Words as IDs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[3]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[4]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">🗺️</div>
            <div><strong>Embeddings</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Learn positions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[5]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[6]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">📊</div>
            <div><strong>Vectors</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Numbers with meaning</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.divider()
    
    # Training corpus
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📚 Training Corpus")
        st.caption("The alien studies this text to learn word meanings")
        
        corpus_text = st.text_area(
            "Training text (one sentence per line):",
            value=DEFAULT_CORPUS,
            height=200,
            label_visibility="collapsed"
        )
    
    # Get embedder
    embedder = get_embedder(strategy, corpus_text, dimensions)
    
    with col2:
        st.markdown("## 📊 Training Stats")
        
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("📚 Vocabulary", f"{embedder.vocab_size()} words")
        with stats_cols[1]:
            st.metric("📐 Dimensions", f"{dimensions}D")
        with stats_cols[2]:
            st.metric("📈 Strategy", strategy.capitalize())
        
        # Strategy description
        if strategy == "random":
            st.warning("⚠️ **Random** embeddings have NO real meaning - just a baseline!")
        elif strategy == "cooccurrence":
            st.info("🔗 **Co-occurrence** learns from which words appear together")
        else:
            st.success("🎯 **Prediction** learns by predicting context words (like Word2Vec)")
    
    st.divider()
    
    # Word Explorer Section
    st.markdown("## 🔍 Word Explorer")
    
    explore_cols = st.columns([1, 2])
    
    with explore_cols[0]:
        # Get list of words for the selectbox
        all_words = embedder.get_all_words()
        
        # Default word
        default_word = "king" if "king" in all_words else all_words[0] if all_words else ""
        
        explore_word = st.text_input(
            "Enter a word to explore:",
            value=default_word,
            help="Type any word from the corpus"
        ).lower().strip()
    
    with explore_cols[1]:
        if explore_word:
            if explore_word in embedder.vocab:
                # Get vector
                vector = embedder.get_vector(explore_word)
                
                # Display vector (first 10 dimensions)
                st.markdown(f"### Vector for '{explore_word}' (first 10 dims)")
                
                # Create colored boxes for vector values
                vector_html = ""
                for i, val in enumerate(vector[:10]):
                    if val > 0.1:
                        css_class = "vector-positive"
                    elif val < -0.1:
                        css_class = "vector-negative"
                    else:
                        css_class = "vector-neutral"
                    vector_html += f'<span class="vector-box {css_class}">{val:.2f}</span>'
                
                if len(vector) > 10:
                    vector_html += f'<span class="vector-box vector-neutral">...+{len(vector)-10} more</span>'
                
                st.markdown(f'<div style="line-height: 2.5;">{vector_html}</div>', unsafe_allow_html=True)
                
                # Similar words
                st.markdown("### 🎯 Most Similar Words")
                similar = embedder.most_similar(explore_word, top_n=5)
                
                for word, score in similar:
                    # Create progress bar for similarity
                    bar_width = max(0, min(100, int(score * 100)))
                    color = "#10b981" if score > 0.5 else "#f59e0b" if score > 0.2 else "#6b7280"
                    st.markdown(f"""
                    <div style="margin: 5px 0;">
                        <span style="display: inline-block; width: 100px; font-weight: bold;">{word}</span>
                        <span style="display: inline-block; width: 60px;">{score:.3f}</span>
                        <div style="display: inline-block; width: 200px; height: 15px; background: #1f2937; border-radius: 3px;">
                            <div style="width: {bar_width}%; height: 100%; background: {color}; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Word '{explore_word}' not in vocabulary!")
                st.caption(f"Try one of: {', '.join(all_words[:10])}...")
    
    st.divider()
    
    # Analogy Solver
    st.markdown("## ✨ Analogy Solver")
    st.markdown("*Solve: A is to B as C is to ?*")
    
    analogy_cols = st.columns(4)
    
    with analogy_cols[0]:
        word_a = st.text_input("A (e.g., king)", value="king", key="analogy_a").lower().strip()
    
    with analogy_cols[1]:
        word_b = st.text_input("B (e.g., queen)", value="queen", key="analogy_b").lower().strip()
    
    with analogy_cols[2]:
        word_c = st.text_input("C (e.g., man)", value="man", key="analogy_c").lower().strip()
    
    with analogy_cols[3]:
        st.markdown("### = ?")
        if word_a and word_b and word_c:
            # Check if all words are in vocab
            missing = [w for w in [word_a, word_b, word_c] if w not in embedder.vocab]
            if missing:
                st.error(f"❌ Unknown: {', '.join(missing)}")
            else:
                try:
                    results = embedder.analogy(word_a, word_b, word_c, top_n=3)
                    for i, (word, score) in enumerate(results):
                        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                        st.markdown(f"{emoji} **{word}** ({score:.2f})")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Show the math
    if word_a and word_b and word_c and all(w in embedder.vocab for w in [word_a, word_b, word_c]):
        st.markdown(f"""
        <div class="info-card">
            <strong>📐 Vector Math:</strong> {word_b} - {word_a} + {word_c} = ?<br/>
            <em>The direction from "{word_a}" to "{word_b}" applied to "{word_c}"</em>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 2D Visualization
    st.markdown("## 🎨 2D Word Map")
    st.caption("Words projected to 2D using PCA - similar words cluster together!")
    
    # Get all embeddings and project to 2D
    if embedder.embeddings is not None and len(embedder.vocab) > 2:
        # Simple PCA for 2D projection
        from numpy.linalg import svd
        
        # Center the data
        embeddings = embedder.embeddings
        centered = embeddings - embeddings.mean(axis=0)
        
        # SVD for PCA
        try:
            U, S, Vt = svd(centered, full_matrices=False)
            coords_2d = U[:, :2] * S[:2]
            
            # Normalize to 0-1 range for plotting
            coords_2d[:, 0] = (coords_2d[:, 0] - coords_2d[:, 0].min()) / (coords_2d[:, 0].max() - coords_2d[:, 0].min() + 1e-10)
            coords_2d[:, 1] = (coords_2d[:, 1] - coords_2d[:, 1].min()) / (coords_2d[:, 1].max() - coords_2d[:, 1].min() + 1e-10)
            
            # Word categories for coloring
            royalty_words = {"king", "queen", "prince", "princess", "royal", "kingdom", "crown", "castle", "throne"}
            animal_words = {"cat", "dog", "pet", "kitten", "puppy", "mouse", "animal", "animals", "cats", "dogs", "furry"}
            gender_words = {"man", "woman", "boy", "girl", "he", "she"}
            emotion_words = {"happy", "sad", "good", "bad", "joy", "happiness", "sadness", "unhappiness", "positive", "negative"}
            tech_words = {"machine", "learning", "data", "neural", "networks", "deep", "model", "artificial", "intelligence", "science"}
            
            # Create scatter plot using Streamlit columns and HTML
            import pandas as pd
            
            # Prepare data for chart
            chart_data = []
            for idx, word in embedder.inverse_vocab.items():
                x, y = coords_2d[idx]
                
                # Determine category
                if word in royalty_words:
                    category = "👑 Royalty"
                elif word in animal_words:
                    category = "🐾 Animals"
                elif word in gender_words:
                    category = "👤 Gender"
                elif word in emotion_words:
                    category = "😊 Emotions"
                elif word in tech_words:
                    category = "💻 Tech"
                else:
                    category = "📝 Other"
                
                chart_data.append({
                    "word": word,
                    "x": float(x),
                    "y": float(y),
                    "category": category
                })
            
            df = pd.DataFrame(chart_data)
            
            # Display using Streamlit's built-in scatter chart
            st.scatter_chart(
                df,
                x="x",
                y="y",
                color="category",
                size=20,
                height=400
            )
            
            # Show word list by category
            with st.expander("📋 Words by Category"):
                cat_cols = st.columns(5)
                categories = ["👑 Royalty", "🐾 Animals", "👤 Gender", "😊 Emotions", "💻 Tech"]
                for i, cat in enumerate(categories):
                    with cat_cols[i]:
                        words_in_cat = df[df["category"] == cat]["word"].tolist()
                        st.markdown(f"**{cat}**")
                        st.caption(", ".join(words_in_cat[:8]))
                        
        except Exception as e:
            st.warning(f"Could not create 2D projection: {e}")


# ============================================================================
# TAB 2: How It Works
# ============================================================================
with tab2:
    st.markdown("## 📊 How Embeddings Work")
    
    # Visual explanation of each strategy
    st.markdown("### 🗺️ The Alien's Three Mapping Methods")
    
    method_cols = st.columns(3)
    
    with method_cols[0]:
        st.markdown("""
        ### 🎲 Random
        **Analogy:** Throwing darts blindfolded
        
        ```python
        # Just random numbers!
        "king" → [0.23, -0.71, 0.42, ...]
        "queen" → [-0.15, 0.89, -0.33, ...]
        # No relation! 😕
        ```
        
        **Reality:**
        - ❌ No semantic meaning
        - ❌ Random similarities
        - ❌ Analogies don't work
        
        *Baseline to show learning matters!*
        """)
    
    with method_cols[1]:
        st.markdown("""
        ### 🔗 Co-occurrence
        **Analogy:** "You are the company you keep"
        
        ```python
        # Words in same context → similar
        "king" appears with: throne, crown
        "queen" appears with: throne, crown
        # Similar contexts → close vectors!
        ```
        
        **How:**
        1. Count word pairs in windows
        2. Apply PPMI transform
        3. Reduce with SVD
        
        ✅ Fast, captures topic similarity
        """)
    
    with method_cols[2]:
        st.markdown("""
        ### 🎯 Prediction
        **Analogy:** Learning by guessing
        
        ```python
        # Given "king", predict neighbors
        "The ___ sits on throne"
        → Learn that king, queen work here!
        # Similar predictions → similar vectors
        ```
        
        **How:**
        1. Predict context words
        2. Adjust vectors via gradient descent
        3. Similar predictions → similar vectors
        
        ✅ Captures nuanced relationships
        """)
    
    st.divider()
    
    # Vector arithmetic explanation
    st.markdown("### ➕ The Magic of Vector Arithmetic")
    
    arith_cols = st.columns([1, 1])
    
    with arith_cols[0]:
        st.markdown("""
        #### How Analogies Work
        
        If embeddings capture meaning, then **directions** encode relationships:
        
        ```
        king  ─────────────→  queen
              "royalty + female"
              
        man   ─────────────→  woman
              "adult + female"
        ```
        
        The direction "male → female" is the same!
        
        So: **king - man + woman ≈ queen** 👑
        """)
    
    with arith_cols[1]:
        st.markdown("""
        #### Famous Examples
        
        | Analogy | Vector Math |
        |---------|-------------|
        | king:queen::man:? | king - man + woman ≈ **queen** |
        | paris:france::berlin:? | paris - france + germany ≈ **berlin** |
        | walking:walk::running:? | walking - walk + run ≈ **running** |
        | cat:kitten::dog:? | cat - kitten + puppy ≈ **dog** |
        
        *Works because the embedding space is structured!*
        """)
    
    st.divider()
    
    # Co-occurrence visualization
    st.markdown("### 🔗 How Co-occurrence Captures Meaning")
    
    cooc_cols = st.columns(2)
    
    with cooc_cols[0]:
        st.markdown("""
        #### The Window Method
        
        For each word, look at its neighbors:
        
        ```
        "The king sat on the throne"
             ↑
           center
        
        Window = 2:
        - Left: "The"
        - Right: "sat", "on"
        
        Count: (king, the)++, (king, sat)++, ...
        ```
        
        Words sharing contexts → similar vectors!
        """)
    
    with cooc_cols[1]:
        st.markdown("""
        #### PPMI: Meaningful Co-occurrence
        
        Raw counts aren't enough:
        - "the" appears with EVERYTHING
        - Doesn't mean much!
        
        **PPMI** measures: "Do these words appear together MORE than chance?"
        
        ```
        PMI(king, crown) = HIGH  ✅
        PMI(king, the) = LOW     ❌
        ```
        
        *Pointwise Mutual Information filters noise!*
        """)
    
    st.divider()
    
    # Why this matters for LLMs
    st.markdown("### 🧠 Why This Matters for LLMs")
    
    llm_cols = st.columns(2)
    
    with llm_cols[0]:
        st.markdown("""
        #### From Tokens to Understanding
        
        Workshop 1 gave us **Token IDs**:
        ```
        "The king" → [42, 891]
        ```
        
        But 891 is just a number - no meaning!
        
        Embeddings give meaning:
        ```
        891 → [0.2, -0.5, 0.8, ...]
               ↑ captures "royalty"
        ```
        
        **This is the LLM's "understanding" of words!**
        """)
    
    with llm_cols[1]:
        st.markdown("""
        #### Real-World Embeddings
        
        | Model | Embedding Dim | Vocab Size |
        |-------|--------------|------------|
        | Word2Vec | 300 | 3M words |
        | GloVe | 300 | 2.2M words |
        | GPT-4 | 12,288 | ~100K tokens |
        | BERT | 768 | ~30K tokens |
        
        **Key insight:** LLMs learn embeddings as part of training!
        
        The embedding layer is literally the first layer of the model.
        """)
    
    st.divider()
    
    st.markdown("### 🔑 Key Takeaways")
    
    st.success("""
    1. **Words become vectors** where position = meaning
    2. **Similar words cluster** in the embedding space  
    3. **Directions encode relationships** (king→queen = man→woman)
    4. **Learning matters** - random embeddings capture nothing!
    """)


# ============================================================================
# TAB 3: Compare Strategies
# ============================================================================
with tab3:
    st.markdown("## 🆚 Strategy Comparison")
    
    st.markdown("""
    ### 🎯 The Key Question: Does Learning Matter?
    
    Let's compare how each strategy handles semantic similarity.
    Random embeddings are like a "cheating" baseline - if they work, no real learning happened!
    """)
    
    # Test word input
    test_word = st.text_input(
        "Enter a word to compare across strategies:",
        value="king",
        help="Choose a word to see how each strategy finds similar words"
    ).lower().strip()
    
    # Get all three embedders
    random_emb = get_embedder("random", corpus_text, dimensions)
    cooc_emb = get_embedder("cooccurrence", corpus_text, dimensions)
    pred_emb = get_embedder("prediction", corpus_text, dimensions)
    
    if test_word:
        if test_word not in random_emb.vocab:
            st.error(f"❌ Word '{test_word}' not in vocabulary!")
            st.caption(f"Try one of: {', '.join(list(random_emb.vocab.keys())[:15])}...")
        else:
            st.markdown("### 📊 Similar Words Comparison")
            
            compare_cols = st.columns(3)
            
            with compare_cols[0]:
                st.markdown("#### 🎲 Random")
                st.caption("Baseline - no real meaning")
                
                try:
                    similar = random_emb.most_similar(test_word, top_n=5)
                    for word, score in similar:
                        st.markdown(f"• **{word}** ({score:.2f})")
                except Exception as e:
                    st.error(str(e))
                
                st.markdown("---")
                st.markdown("❌ Random matches!")
            
            with compare_cols[1]:
                st.markdown("#### 🔗 Co-occurrence")
                st.caption("Context-based learning")
                
                try:
                    similar = cooc_emb.most_similar(test_word, top_n=5)
                    for word, score in similar:
                        st.markdown(f"• **{word}** ({score:.2f})")
                except Exception as e:
                    st.error(str(e))
                
                st.markdown("---")
                st.markdown("✅ Semantic matches!")
            
            with compare_cols[2]:
                st.markdown("#### 🎯 Prediction")
                st.caption("Neural learning")
                
                try:
                    similar = pred_emb.most_similar(test_word, top_n=5)
                    for word, score in similar:
                        st.markdown(f"• **{word}** ({score:.2f})")
                except Exception as e:
                    st.error(str(e))
                
                st.markdown("---")
                st.markdown("✅ Learned relationships!")
            
            st.divider()
            
            # Similarity test between word pairs
            st.markdown("### 📏 Similarity Scores Comparison")
            
            # Test pairs
            if test_word == "king":
                test_pairs = [("king", "queen"), ("king", "man"), ("king", "cat"), ("king", "happy")]
            elif test_word == "cat":
                test_pairs = [("cat", "dog"), ("cat", "kitten"), ("cat", "king"), ("cat", "happy")]
            elif test_word == "happy":
                test_pairs = [("happy", "sad"), ("happy", "good"), ("happy", "joy"), ("happy", "king")]
            else:
                # Generate pairs dynamically
                other_words = [w for w in list(random_emb.vocab.keys())[:10] if w != test_word]
                test_pairs = [(test_word, w) for w in other_words[:4]]
            
            # Filter to only include valid pairs
            valid_pairs = [(w1, w2) for w1, w2 in test_pairs if w1 in random_emb.vocab and w2 in random_emb.vocab]
            
            if valid_pairs:
                import pandas as pd
                
                data = []
                for w1, w2 in valid_pairs:
                    try:
                        row = {
                            "Word Pair": f"{w1} / {w2}",
                            "🎲 Random": f"{random_emb.similarity(w1, w2):.3f}",
                            "🔗 Co-occur": f"{cooc_emb.similarity(w1, w2):.3f}",
                            "🎯 Predict": f"{pred_emb.similarity(w1, w2):.3f}",
                        }
                        data.append(row)
                    except:
                        pass
                
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Analogy test
            st.markdown("### ✨ Analogy Test")
            
            # Pick relevant analogies
            analogy_tests = [
                ("king", "queen", "man", "woman"),
                ("cat", "kitten", "dog", "puppy"),
                ("happy", "sad", "good", "bad"),
            ]
            
            for a, b, c, expected in analogy_tests:
                if all(w in random_emb.vocab for w in [a, b, c]):
                    st.markdown(f"**{a} : {b} :: {c} : ?** (expected: {expected})")
                    
                    result_cols = st.columns(3)
                    
                    with result_cols[0]:
                        try:
                            result = random_emb.analogy(a, b, c, top_n=1)
                            answer = result[0][0] if result else "?"
                            emoji = "❌" if answer != expected else "✅"
                            st.markdown(f"🎲 Random: **{answer}** {emoji}")
                        except:
                            st.markdown("🎲 Random: Error")
                    
                    with result_cols[1]:
                        try:
                            result = cooc_emb.analogy(a, b, c, top_n=1)
                            answer = result[0][0] if result else "?"
                            emoji = "✅" if answer == expected else "🔶"
                            st.markdown(f"🔗 Co-occur: **{answer}** {emoji}")
                        except:
                            st.markdown("🔗 Co-occur: Error")
                    
                    with result_cols[2]:
                        try:
                            result = pred_emb.analogy(a, b, c, top_n=1)
                            answer = result[0][0] if result else "?"
                            emoji = "✅" if answer == expected else "🔶"
                            st.markdown(f"🎯 Predict: **{answer}** {emoji}")
                        except:
                            st.markdown("🎯 Predict: Error")
                    
                    st.markdown("")
    
    st.divider()
    
    # Key insight table
    st.markdown("### 💡 Strategy Comparison Table")
    
    import pandas as pd
    
    comparison_data = pd.DataFrame({
        "Aspect": [
            "Semantic Meaning",
            "Similar Words",
            "Analogies",
            "Speed",
            "Data Needed",
            "Use Case"
        ],
        "🎲 Random": [
            "❌ None",
            "❌ Random",
            "❌ Fails",
            "✅ Instant",
            "✅ None",
            "Baseline only"
        ],
        "🔗 Co-occurrence": [
            "✅ Good",
            "✅ Topical",
            "🔶 Sometimes",
            "✅ Fast",
            "🔶 Medium",
            "Topic similarity"
        ],
        "🎯 Prediction": [
            "✅ Best",
            "✅ Semantic",
            "✅ Works",
            "🔶 Slower",
            "❌ More data",
            "NLP tasks"
        ]
    })
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### 🔑 Key Insight
    
    > **Random embeddings are "cheating"** - they look like they work (similar API, same dimensions) 
    > but capture NO semantic meaning. They're our control group to prove that **learning matters**!
    
    - **Co-occurrence** captures that words in similar contexts have similar meanings
    - **Prediction** learns even more nuanced relationships through neural training
    - **Random** is just noise - if random works for your task, you don't need embeddings!
    """)


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🗺️ Workshop 2: Embeddings | GenAI Self-Build Series</p>
    <p><em>Remember: The alien (algorithm) creates a map where similar words live close together!</em></p>
</div>
""", unsafe_allow_html=True)
