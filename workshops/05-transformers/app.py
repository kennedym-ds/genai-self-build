"""
🧠 Workshop 5: Transformer Explorer
====================================

Interactive Streamlit app to visualize transformer architecture.

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from transformer import (
    softmax, layer_norm, gelu,
    PositionalEncoding, MultiHeadAttention, FeedForward,
    TransformerBlock, MiniTransformer, SimpleVocab
)


# Page config
st.set_page_config(
    page_title="🧠 Transformer Explorer",
    page_icon="🧠",
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
    .token-box {
        display: inline-block;
        padding: 8px 12px;
        margin: 4px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 1rem;
        text-align: center;
    }
    .embed-token { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .pos-token { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .combined-token { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .output-token { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }
    
    .component-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #4a4a6a;
    }
    .component-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .layer-box {
        background: rgba(100, 100, 200, 0.1);
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .attention-head {
        background: rgba(240, 147, 251, 0.1);
        border: 1px solid #f093fb;
        border-radius: 6px;
        padding: 10px;
        margin: 5px;
        display: inline-block;
    }
    .ffn-box {
        background: rgba(67, 233, 123, 0.1);
        border: 1px solid #43e97b;
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
    }
    .pipeline-arrow {
        font-size: 2rem;
        color: #667eea;
        text-align: center;
        margin: 10px 0;
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
    .prob-bar {
        height: 20px;
        border-radius: 4px;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Transformer Explorer")
st.markdown("### *Watch the alien's brain process language, step by step*")

# Sidebar
with st.sidebar:
    st.markdown("## 🛸 The Factory Analogy")
    st.markdown("""
    Think of a transformer as an **alien's language factory**:
    
    1. 📦 **Raw Materials** (Input Embedding)
       - Words converted to number vectors
    
    2. 📍 **Position Stamps** (Positional Encoding)  
       - Each word gets a position marker
    
    3. 👀 **Quality Control** (Attention)
       - Workers check relationships between parts
       - Multiple inspection teams (heads)
    
    4. 🔧 **Processing** (Feed-Forward)
       - Transform and refine the information
    
    5. 🔄 **Repeat** (Stacked Layers)
       - Multiple rounds of inspection + processing
    
    6. 📤 **Output** (Final Prediction)
       - Decide what word comes next
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    embed_dim = st.select_slider(
        "Embedding Dimension",
        options=[32, 64, 128],
        value=64
    )
    
    num_heads = st.select_slider(
        "Number of Attention Heads",
        options=[2, 4, 8],
        value=4
    )
    
    num_layers = st.select_slider(
        "Number of Layers",
        options=[1, 2, 4],
        value=2
    )
    
    st.markdown("---")
    st.markdown("### 📊 Workshop 5 of 6")


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Generate Text",
    "🏭 Architecture Walkthrough", 
    "👀 Attention Visualization",
    "📊 Component Inspector"
])


# Tab 1: Text Generation
with tab1:
    st.markdown("## 🔬 Interactive Text Generation")
    st.markdown("""
    Watch the transformer predict the next word, step by step. 
    This is a **decoder-only** transformer (like GPT) - it predicts what comes next.
    """)
    
    # Important notice about untrained model
    st.warning("""
    ⚠️ **Important: This model is UNTRAINED!**
    
    The output will look random/nonsensical because:
    - 🎲 **Random weights** — The model was just initialized, never trained
    - 📚 **No learning** — Real LLMs train on trillions of tokens for weeks
    - 🧠 **No patterns learned** — It doesn't know grammar, facts, or logic yet
    
    **This demo shows the ARCHITECTURE working** — data flows correctly through 
    embeddings → attention → FFN → predictions. The magic of coherent text comes 
    from training, which requires massive compute we don't have here.
    
    *Think of it like a car engine: mechanically correct, but without fuel (training data) 
    and ignition (gradient descent), it won't drive anywhere meaningful!*
    """)
    
    # Initialize model and vocab
    @st.cache_resource
    def load_model(vocab_size, embed_dim, num_heads, num_layers):
        np.random.seed(42)  # For reproducibility
        return MiniTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
    
    vocab = SimpleVocab()
    model = load_model(vocab.vocab_size, embed_dim, num_heads, num_layers)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_input(
            "Enter a prompt:",
            value="the cat sat on",
            help="Use simple words from the vocabulary"
        )
    
    with col2:
        num_tokens = st.slider("Tokens to generate:", 1, 10, 5)
        temperature = st.slider("Temperature:", 0.1, 2.0, 1.0, 0.1)
    
    if st.button("🚀 Generate", type="primary"):
        try:
            token_ids = vocab.encode(prompt)
            
            st.markdown("### Generation Process")
            
            # Show initial tokens
            st.markdown("**Input tokens:**")
            cols = st.columns(len(prompt.split()) + num_tokens)
            for i, word in enumerate(prompt.split()):
                with cols[i]:
                    st.markdown(f"<div class='token-box embed-token'>{word}</div>", 
                               unsafe_allow_html=True)
            
            # Generate step by step
            current_ids = token_ids.copy()
            for step in range(num_tokens):
                probs = model.get_next_token_probs(current_ids, temperature=temperature)
                
                # Show top predictions
                top_k = 5
                top_indices = np.argsort(probs)[-top_k:][::-1]
                
                with st.expander(f"Step {step + 1}: Predicting token {len(current_ids) + 1}", 
                                expanded=(step == 0)):
                    st.markdown("**Top 5 predictions:**")
                    for idx in top_indices:
                        word = vocab.id_to_word.get(idx, "<UNK>")
                        prob = probs[idx]
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; margin: 5px 0;'>
                            <div style='width: 80px; font-family: monospace;'>{word}</div>
                            <div class='prob-bar' style='width: {prob*300}px; 
                                background: linear-gradient(90deg, #667eea, #764ba2);'></div>
                            <div style='margin-left: 10px;'>{prob:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Sample next token
                next_token = np.random.choice(len(probs), p=probs)
                current_ids.append(next_token)
                
                # Show generated token
                with cols[len(prompt.split()) + step]:
                    next_word = vocab.id_to_word.get(next_token, "<UNK>")
                    st.markdown(f"<div class='token-box output-token'>{next_word}</div>", 
                               unsafe_allow_html=True)
            
            # Final output
            st.markdown("---")
            st.markdown("### 📝 Final Output")
            final_text = vocab.decode(current_ids)
            st.success(f"**{final_text}**")
            
            # Show comparison with trained model
            st.markdown("---")
            st.markdown("### 🆚 Untrained vs Trained Comparison")
            
            comp_cols = st.columns(2)
            with comp_cols[0]:
                st.markdown("""
                <div style='background: #4a1a1a; padding: 15px; border-radius: 10px; border: 2px solid #dc2626;'>
                    <strong>🎲 Our Untrained Model:</strong><br>
                    <code style='color: #fca5a5;'>{}</code><br><br>
                    <small>Random output because weights are random</small>
                </div>
                """.format(final_text), unsafe_allow_html=True)
            
            with comp_cols[1]:
                # Show what a trained model might produce
                trained_examples = {
                    "the cat sat on": "the cat sat on the mat",
                    "the dog": "the dog ran quickly",
                    "a big": "a big red ball",
                    "the bird flew": "the bird flew away",
                }
                expected = trained_examples.get(prompt.lower().strip(), f"{prompt} [coherent continuation]")
                st.markdown("""
                <div style='background: #1a4a1a; padding: 15px; border-radius: 10px; border: 2px solid #16a34a;'>
                    <strong>✅ A Trained Model (like GPT):</strong><br>
                    <code style='color: #86efac;'>{}</code><br><br>
                    <small>Coherent because it learned patterns from data</small>
                </div>
                """.format(expected), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Try using simpler words like: the, cat, dog, sat, on, a, is, and")


# Tab 2: Architecture Walkthrough
with tab2:
    st.markdown("## 🏭 Transformer Architecture")
    st.markdown("""
    Follow data through the transformer factory, from raw words to predictions.
    """)
    
    # Visual pipeline
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #667eea, #764ba2); 
                    border-radius: 10px; color: white;'>
            📝 Input Text
        </div>
        <span style='font-size: 2rem; margin: 0 10px;'>→</span>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #f093fb, #f5576c); 
                    border-radius: 10px; color: white;'>
            🔢 Embeddings + Position
        </div>
        <span style='font-size: 2rem; margin: 0 10px;'>→</span>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #4facfe, #00f2fe); 
                    border-radius: 10px; color: white;'>
            🔄 Transformer Blocks
        </div>
        <span style='font-size: 2rem; margin: 0 10px;'>→</span>
        <div style='display: inline-block; margin: 10px; padding: 15px 25px; 
                    background: linear-gradient(135deg, #43e97b, #38f9d7); 
                    border-radius: 10px; color: white;'>
            📤 Predictions
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step-by-step breakdown
    step_cols = st.columns(2)
    
    with step_cols[0]:
        st.markdown("""
        ### 1️⃣ Token Embedding
        <div class='component-box'>
            <div class='component-title'>📦 Convert Words to Vectors</div>
            <p>Each word becomes a vector of numbers (embedding). 
            Think of it as translating words into a language of numbers 
            that the model understands.</p>
            <p><strong>Shape:</strong> (seq_len, embed_dim)</p>
            <p><strong>Example:</strong> "cat" → [0.2, -0.5, 0.8, ...]</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 2️⃣ Positional Encoding
        <div class='component-box'>
            <div class='component-title'>📍 Add Position Information</div>
            <p>Transformers process all tokens in parallel, so we add 
            position information to tell the model word order.</p>
            <p><strong>Pattern:</strong> Sine/cosine waves at different frequencies</p>
            <p><strong>Why?</strong> "Dog bites man" ≠ "Man bites dog"</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 3️⃣ Attention Mechanism
        <div class='component-box'>
            <div class='component-title'>👀 Relate Tokens to Each Other</div>
            <p>Each token looks at all other tokens (or just previous ones 
            in causal/decoder models) and decides which to focus on.</p>
            <p><strong>Multi-Head:</strong> Multiple attention "perspectives"</p>
            <p><strong>Causal Mask:</strong> Can't look at future tokens</p>
        </div>
        """, unsafe_allow_html=True)
    
    with step_cols[1]:
        st.markdown("""
        ### 4️⃣ Feed-Forward Network
        <div class='component-box'>
            <div class='component-title'>🔧 Transform Information</div>
            <p>After attention gathers context, the FFN processes each 
            position independently with a small neural network.</p>
            <p><strong>Structure:</strong> expand → activate → compress</p>
            <p><strong>Hidden:</strong> Usually 4x the embedding dimension</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 5️⃣ Residual Connections
        <div class='component-box'>
            <div class='component-title'>🔄 Skip Connections</div>
            <p>The original input is added back to the output at each step. 
            This helps gradients flow and preserves original information.</p>
            <p><strong>Formula:</strong> output = LayerNorm(x + sublayer(x))</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 6️⃣ Stacked Layers
        <div class='component-box'>
            <div class='component-title'>📚 Repeat N Times</div>
            <p>Real transformers stack many identical blocks. Each layer 
            builds more abstract understanding on top of the previous.</p>
            <p><strong>GPT-3:</strong> 96 layers</p>
            <p><strong>Our demo:</strong> {num_layers} layers</p>
        </div>
        """.format(num_layers=num_layers), unsafe_allow_html=True)
    
    # Show model stats
    st.markdown("---")
    st.markdown("### 📊 Current Model Configuration")
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{embed_dim}</div>
            <div class='metric-label'>Embedding Dim</div>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[1]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{num_heads}</div>
            <div class='metric-label'>Attention Heads</div>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[2]:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{num_layers}</div>
            <div class='metric-label'>Layers</div>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[3]:
        params = embed_dim * vocab.vocab_size  # embeddings
        params += num_layers * (4 * embed_dim * embed_dim + 2 * embed_dim * 4 * embed_dim)  # approx
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{params//1000}K</div>
            <div class='metric-label'>Parameters</div>
        </div>
        """, unsafe_allow_html=True)


# Tab 3: Attention Visualization
with tab3:
    st.markdown("## 👀 Attention Pattern Visualization")
    st.markdown("""
    See which tokens attend to which other tokens. In a decoder (like GPT), 
    each token can only look at previous tokens (causal masking).
    """)
    
    attn_input = st.text_input(
        "Enter text to analyze:",
        value="the cat sat",
        key="attn_input"
    )
    
    layer_to_show = st.selectbox(
        "Select layer:",
        options=list(range(num_layers)),
        format_func=lambda x: f"Layer {x + 1}"
    )
    
    if st.button("🔍 Analyze Attention", type="primary"):
        vocab = SimpleVocab()
        model = load_model(vocab.vocab_size, embed_dim, num_heads, num_layers)
        
        try:
            token_ids = vocab.encode(attn_input)
            words = attn_input.split()
            
            # Forward pass to get attention weights
            _, all_weights = model.forward(token_ids)
            
            # Get weights for selected layer
            layer_weights = all_weights[layer_to_show]  # (num_heads, seq, seq)
            
            st.markdown(f"### Layer {layer_to_show + 1} Attention Patterns")
            
            # Show each head
            head_cols = st.columns(min(num_heads, 4))
            
            for h in range(num_heads):
                with head_cols[h % 4]:
                    st.markdown(f"**Head {h + 1}**")
                    
                    # Create attention heatmap
                    weights = layer_weights[h]
                    
                    # Display as table
                    import pandas as pd
                    df = pd.DataFrame(
                        weights,
                        index=words,
                        columns=words
                    )
                    
                    # Style the dataframe
                    styled = df.style.background_gradient(cmap='Blues', vmin=0, vmax=1)
                    styled = styled.format("{:.2f}")
                    st.dataframe(styled, use_container_width=True)
            
            # Aggregate view
            st.markdown("---")
            st.markdown("### 📊 Aggregated Attention (Mean Across Heads)")
            
            mean_weights = layer_weights.mean(axis=0)
            
            import pandas as pd
            df_mean = pd.DataFrame(
                mean_weights,
                index=words,
                columns=words
            )
            styled_mean = df_mean.style.background_gradient(cmap='Purples', vmin=0, vmax=1)
            styled_mean = styled_mean.format("{:.2f}")
            st.dataframe(styled_mean, use_container_width=True)
            
            # Explain causal masking
            st.info("""
            📝 **Notice the triangular pattern?** This is **causal masking** - 
            each token can only attend to tokens that came before it 
            (including itself). This is how GPT-style models ensure they 
            don't "cheat" by looking at future tokens during training.
            """)
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Try using simpler words from the vocabulary.")


# Tab 4: Component Inspector
with tab4:
    st.markdown("## 📊 Component Deep Dive")
    st.markdown("""
    Examine each component of the transformer in detail.
    """)
    
    component = st.selectbox(
        "Select component:",
        ["Positional Encoding", "Multi-Head Attention", "Feed-Forward Network", "Full Block"]
    )
    
    if component == "Positional Encoding":
        st.markdown("""
        ### 📍 Positional Encoding Patterns
        
        The transformer uses **sinusoidal positional encodings** - sine and cosine 
        waves at different frequencies. Lower dimensions have higher frequencies 
        (vary more between adjacent positions).
        """)
        
        pos_enc = PositionalEncoding(embed_dim=embed_dim, max_seq_len=100)
        encodings = pos_enc(50)
        
        # Show heatmap
        import pandas as pd
        
        # Show first 16 dimensions for clarity
        show_dims = min(16, embed_dim)
        df = pd.DataFrame(
            encodings[:, :show_dims],
            index=[f"Pos {i}" for i in range(50)],
            columns=[f"Dim {i}" for i in range(show_dims)]
        )
        
        st.markdown("**Encoding values (first 16 dimensions, 50 positions):**")
        styled = df.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1)
        styled = styled.format("{:.2f}")
        st.dataframe(styled, use_container_width=True, height=400)
        
        st.info("""
        🔍 **What to notice:**
        - Odd dimensions (1, 3, 5...) use cosine, even dimensions (0, 2, 4...) use sine
        - Lower dimensions oscillate faster (vary more between positions)
        - Higher dimensions oscillate slower (encode longer-range position info)
        - Every position has a unique "fingerprint"
        """)
    
    elif component == "Multi-Head Attention":
        st.markdown("""
        ### 👀 Multi-Head Attention
        
        Instead of one attention mechanism, we use multiple "heads" that each 
        learn to focus on different types of relationships.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Why multiple heads?**
            - Head 1 might focus on syntax (subject-verb)
            - Head 2 might focus on coreference (pronouns)
            - Head 3 might focus on proximity (nearby words)
            - Head 4 might focus on semantics (related concepts)
            """)
        
        with col2:
            st.markdown(f"""
            **Current configuration:**
            - Embedding dimension: {embed_dim}
            - Number of heads: {num_heads}
            - Head dimension: {embed_dim // num_heads}
            - Total attention parameters: ~{4 * embed_dim * embed_dim:,}
            """)
        
        st.markdown("---")
        st.markdown("**Try it:**")
        
        test_input = st.text_input("Enter test sequence:", "hello world test", key="mha_test")
        
        if st.button("Run Attention"):
            vocab = SimpleVocab()
            attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
            # Create random embeddings for demo
            seq_len = len(test_input.split())
            x = np.random.randn(seq_len, embed_dim).astype(np.float32) * 0.1
            
            output, weights = attn(x, causal=True)
            
            st.markdown(f"**Input shape:** ({seq_len}, {embed_dim})")
            st.markdown(f"**Output shape:** ({output.shape[0]}, {output.shape[1]})")
            st.markdown(f"**Attention weights shape:** ({weights.shape[0]}, {weights.shape[1]}, {weights.shape[2]})")
            
            st.success("✅ Attention computed successfully!")
    
    elif component == "Feed-Forward Network":
        st.markdown("""
        ### 🔧 Feed-Forward Network (FFN)
        
        The FFN processes each position independently after attention has mixed 
        information across positions.
        """)
        
        ffn = FeedForward(embed_dim=embed_dim)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Architecture:**
            1. Linear: {embed_dim} → {ffn.hidden_dim}
            2. GELU activation
            3. Linear: {ffn.hidden_dim} → {embed_dim}
            
            **Why expand-then-compress?**
            - More capacity to learn complex transformations
            - GELU adds non-linearity (ability to learn curves)
            - Standard ratio is 4x expansion
            """)
        
        with col2:
            st.markdown("**GELU Activation:**")
            x = np.linspace(-3, 3, 100)
            y = gelu(x)
            
            import pandas as pd
            chart_data = pd.DataFrame({"x": x, "GELU(x)": y})
            st.line_chart(chart_data.set_index("x"), height=200)
            
            st.caption("GELU is smoother than ReLU, with slight negative values allowed")
    
    else:  # Full Block
        st.markdown("""
        ### 🧱 Transformer Block
        
        One complete transformer block combines:
        1. Multi-Head Self-Attention
        2. Add & Layer Norm (residual)
        3. Feed-Forward Network
        4. Add & Layer Norm (residual)
        """)
        
        st.markdown("""
        ```
        Input x
            │
            ├──────────────────────┐
            │                      │ (residual)
            ▼                      │
        Multi-Head Attention       │
            │                      │
            ▼                      │
            + ◄────────────────────┘
            │
            ▼
        Layer Norm
            │
            ├──────────────────────┐
            │                      │ (residual)
            ▼                      │
        Feed-Forward Network       │
            │                      │
            ▼                      │
            + ◄────────────────────┘
            │
            ▼
        Layer Norm
            │
            ▼
        Output
        ```
        """)
        
        st.info("""
        🔑 **Key insight:** Residual connections (the `+` operations) are crucial!
        They allow gradients to flow directly back through the network and help 
        preserve the original signal. Without them, deep transformers wouldn't train.
        """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    🧠 Workshop 5: Transformers | Part of the GenAI Self-Build Series (5 of 6)
</div>
""", unsafe_allow_html=True)
