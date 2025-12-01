"""
🎯 Workshop 4: Attention Mechanism - Interactive Demo
=====================================================

A Streamlit app for exploring how attention works in transformers.

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from typing import List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import SimpleAttention, softmax


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Workshop 4: Attention Mechanism",
    page_icon="👀",
    layout="wide"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .analogy-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #8b5cf6;
    }
    .token-box {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 8px;
        font-family: monospace;
        font-weight: bold;
    }
    .attention-high { background: linear-gradient(135deg, #dc2626, #f87171); color: white; }
    .attention-medium { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: black; }
    .attention-low { background: linear-gradient(135deg, #3b82f6, #60a5fa); color: white; }
    .attention-none { background: linear-gradient(135deg, #374151, #4b5563); color: #9ca3af; }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #4b5563;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #8b5cf6;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
    }
    .flow-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        border: 1px solid #334155;
    }
    .flow-step {
        display: inline-block;
        padding: 12px 20px;
        margin: 8px;
        border-radius: 8px;
        font-weight: bold;
    }
    .flow-input { background: #3b82f6; color: white; }
    .flow-process { background: #8b5cf6; color: white; }
    .flow-output { background: #10b981; color: white; }
    .masked-token {
        background: #1f2937;
        color: #4b5563;
        text-decoration: line-through;
    }
    .causal-allowed { background: #10b981; color: white; }
    .causal-blocked { background: #dc2626; color: white; opacity: 0.5; }
    
    /* Fix table overlap issues */
    .attention-table-container {
        overflow-x: auto;
        margin: 15px 0;
        padding: 10px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        display: block;
        width: 100%;
    }
    .attention-table-container table {
        border-collapse: separate;
        border-spacing: 4px;
        width: auto;
        min-width: 100%;
        table-layout: fixed;
    }
    .attention-table-container td {
        min-width: 50px;
        max-width: 80px;
        word-wrap: break-word;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .attention-table-container h4 {
        margin: 0 0 10px 0;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_word_relationships(words: List[str]) -> np.ndarray:
    """
    Create an explicit attention bias matrix based on linguistic relationships.
    
    This directly encodes what attention patterns SHOULD look like for educational purposes.
    Returns a matrix where higher values = stronger attention relationship.
    """
    n = len(words)
    words_lower = [w.lower() for w in words]
    
    # Start with very small base attention
    relationships = np.ones((n, n)) * 0.02
    
    # Self-attention is low-moderate
    for i in range(n):
        relationships[i, i] = 0.15
    
    # Define relationship rules
    pronouns = {'it', 'he', 'she', 'they', 'him', 'her', 'them', 'who', 'which', 'that'}
    animate = {'dog', 'cat', 'man', 'woman', 'boy', 'girl', 'doctor', 'nurse', 'student', 
               'teacher', 'chef', 'programmer', 'ceo', 'professor', 'alien', 'knight',
               'john', 'mary', 'person', 'user', 'children'}
    objects = {'book', 'ball', 'trophy', 'suitcase', 'table', 'cabinet', 'keys', 'code',
               'bug', 'fish', 'pasta', 'award', 'capital', 'company', 'weather', 'patterns',
               'tomatoes', 'basil', 'park', 'mat'}
    verbs = {'chased', 'ran', 'gave', 'told', 'asked', 'saw', 'left', 'ate', 'fixed',
             'bought', 'cooked', 'thanked', 'praised', 'said', 'learned', 'studied',
             'received', 'sat', 'is', 'are', 'was', 'were', 'would', 'will', 'fit'}
    adjectives = {'quick', 'big', 'small', 'hungry', 'fresh', 'delicious', 'brave', 
                  'tired', 'fast', 'lazy', 'brown'}
    
    for i, word_i in enumerate(words_lower):
        for j, word_j in enumerate(words_lower):
            if i == j:
                continue
                
            # RULE 1: Pronouns attend VERY STRONGLY to nearby nouns
            if word_i in pronouns:
                if word_j in animate or word_j in objects:
                    distance = abs(i - j)
                    # Very strong attention to nouns, especially closer ones
                    relationships[i, j] = max(relationships[i, j], 0.95 - distance * 0.08)
            
            # RULE 2: Verbs attend strongly to their likely subjects (animate nouns before)
            if word_i in verbs:
                if word_j in animate and j < i:
                    distance = i - j
                    relationships[i, j] = max(relationships[i, j], 0.85 - distance * 0.1)
            
            # RULE 3: Verbs attend to their likely objects (nouns after them)
            if word_i in verbs:
                if (word_j in objects or word_j in animate) and j > i:
                    distance = j - i
                    relationships[i, j] = max(relationships[i, j], 0.70 - distance * 0.1)
            
            # RULE 4: Adjectives attend strongly to nearby nouns
            if word_i in adjectives:
                if word_j in animate or word_j in objects:
                    distance = abs(i - j)
                    if distance <= 2:
                        relationships[i, j] = max(relationships[i, j], 0.80)
            
            # RULE 5: Nouns attend to their modifying adjectives
            if word_i in (animate | objects):
                if word_j in adjectives and abs(i - j) <= 2:
                    relationships[i, j] = max(relationships[i, j], 0.60)
            
            # RULE 6: "the" attends strongly to the noun it modifies
            if word_i == 'the':
                if word_j in (animate | objects | adjectives):
                    distance = j - i
                    if 0 < distance <= 3:
                        relationships[i, j] = max(relationships[i, j], 0.75 - distance * 0.1)
            
            # RULE 7: Same word = moderate attention
            if word_i == word_j:
                relationships[i, j] = max(relationships[i, j], 0.40)
            
            # RULE 8: Connectors attend moderately to verbs and nouns
            if word_i in {'and', 'because', 'but', 'so', 'that', 'which', 'who'}:
                if word_j in verbs or word_j in animate:
                    relationships[i, j] = max(relationships[i, j], 0.35)
    
    return relationships


def create_word_embeddings(words: List[str], dim: int = 64, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create embeddings that will produce meaningful attention patterns.
    
    Returns both embeddings AND a relationship bias matrix.
    """
    np.random.seed(seed)
    
    # Create random base embeddings (these will be transformed)
    embeddings = np.random.randn(len(words), dim).astype(np.float32)
    
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Get the explicit relationship matrix
    relationships = get_word_relationships(words)
    
    return embeddings, relationships


def compute_attention_for_demo(
    words: List[str], 
    query_idx: int, 
    attention_type: str = "scaled",
    num_heads: int = 1,
    use_causal: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute attention weights for demonstration with meaningful patterns."""
    dim = 64
    embeddings, relationship_bias = create_word_embeddings(words, dim)
    
    attn = SimpleAttention(embed_dim=dim, num_heads=num_heads)
    
    # Compute base attention
    if use_causal:
        mask = SimpleAttention.create_causal_mask(len(words))
        output, weights = attn.self_attention(embeddings, mask=mask)
    elif attention_type == "dot_product":
        output, weights = attn.dot_product_attention(embeddings, embeddings, embeddings)
    elif attention_type == "scaled":
        output, weights = attn.scaled_dot_product_attention(embeddings, embeddings, embeddings)
    else:  # multi_head
        output, weights = attn.multi_head_attention(embeddings, embeddings, embeddings)
    
    # Blend the computed attention with our relationship bias for educational clarity
    # This ensures patterns are visible while still showing attention mechanism behavior
    if len(weights.shape) == 2:
        # Single head: blend 15% computed + 85% relationships for clear patterns
        weights = 0.15 * weights + 0.85 * relationship_bias
        # Re-normalize rows to sum to 1
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
        
        # Apply causal mask if needed
        if use_causal:
            mask = np.triu(np.ones_like(weights), k=1)
            weights = np.where(mask, 0, weights)
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    else:
        # Multi-head: apply different relationship emphasis to each head
        for h in range(weights.shape[0]):
            # Each head gets a slightly different blend for variety
            head_bias = relationship_bias * (0.7 + 0.3 * np.random.rand())
            weights[h] = 0.15 * weights[h] + 0.85 * head_bias
            weights[h] = weights[h] / (weights[h].sum(axis=1, keepdims=True) + 1e-8)
    
    return output, weights


def render_attention_heatmap(words: List[str], weights: np.ndarray, title: str = "Attention Weights"):
    """Render attention weights as a colored heatmap."""
    # Handle multi-head case - average across heads
    if len(weights.shape) == 3:
        weights = weights.mean(axis=0)
    
    n = len(words)
    
    # Truncate long words for display
    display_words = [w[:8] + "…" if len(w) > 8 else w for w in words]
    
    # Create heatmap HTML wrapped in container
    html = f"<div class='attention-table-container'>"
    html += f"<h4>{title}</h4>"
    html += "<table>"
    
    # Header row
    html += "<tr><td style='padding: 6px; min-width: 60px;'></td>"
    for w in display_words:
        html += f"<td style='padding: 6px; text-align: center; font-weight: bold; color: #9ca3af; font-size: 0.85rem;'>{w}</td>"
    html += "</tr>"
    
    # Data rows
    for i, query_word in enumerate(display_words):
        html += f"<tr><td style='padding: 6px; font-weight: bold; color: #9ca3af; font-size: 0.85rem; white-space: nowrap;'>{query_word}</td>"
        for j in range(n):
            weight = weights[i, j]
            # Color based on attention weight
            if weight > 0.3:
                bg_color = f"rgba(239, 68, 68, {min(weight * 1.5, 1)})"  # Red
                text_color = "white"
            elif weight > 0.15:
                bg_color = f"rgba(245, 158, 11, {min(weight * 2, 1)})"  # Amber
                text_color = "black"
            elif weight > 0.05:
                bg_color = f"rgba(59, 130, 246, {min(weight * 3, 1)})"  # Blue
                text_color = "white"
            else:
                bg_color = "rgba(75, 85, 99, 0.3)"  # Gray
                text_color = "#9ca3af"
            
            html += f"<td style='padding: 6px; text-align: center; background: {bg_color}; color: {text_color}; border-radius: 4px; font-size: 0.8rem;'>{weight:.2f}</td>"
        html += "</tr>"
    
    html += "</table>"
    html += "</div>"
    return html


def render_attention_visualization(words: List[str], weights: np.ndarray, query_idx: int):
    """Render attention as spotlights on words."""
    # Handle multi-head case
    if len(weights.shape) == 3:
        weights = weights.mean(axis=0)
    
    query_weights = weights[query_idx]
    
    html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; align-items: center;'>"
    
    for i, (word, weight) in enumerate(zip(words, query_weights)):
        # Determine spotlight intensity
        if i == query_idx:
            # Query word
            style = "background: linear-gradient(135deg, #8b5cf6, #a78bfa); color: white; border: 3px solid #c4b5fd;"
        elif weight > 0.25:
            # High attention
            style = f"background: linear-gradient(135deg, #dc2626, #f87171); color: white; box-shadow: 0 0 {int(weight * 30)}px rgba(220, 38, 38, 0.6);"
        elif weight > 0.15:
            # Medium attention
            style = f"background: linear-gradient(135deg, #f59e0b, #fbbf24); color: black; box-shadow: 0 0 {int(weight * 20)}px rgba(245, 158, 11, 0.5);"
        elif weight > 0.08:
            # Low attention
            style = f"background: linear-gradient(135deg, #3b82f6, #60a5fa); color: white; box-shadow: 0 0 {int(weight * 15)}px rgba(59, 130, 246, 0.4);"
        else:
            # Minimal attention
            style = "background: #374151; color: #6b7280;"
        
        html += f"""
        <div style='padding: 10px 16px; border-radius: 8px; font-family: monospace; font-size: 1.1rem; {style}'>
            {word}
            <div style='font-size: 0.7rem; opacity: 0.8;'>{weight:.1%}</div>
        </div>
        """
    
    html += "</div>"
    return html


def render_causal_mask_visual(words: List[str]):
    """Visualize causal masking."""
    n = len(words)
    
    # Truncate long words for display
    display_words = [w[:8] + "…" if len(w) > 8 else w for w in words]
    
    html = "<div class='attention-table-container'>"
    html += "<h4>🎭 Causal Mask (GPT-style)</h4>"
    html += "<p style='color: #9ca3af; margin-bottom: 10px;'>Each word can only attend to itself and previous words (green = allowed, red = blocked)</p>"
    html += "<table>"
    
    # Header
    html += "<tr><td style='padding: 6px; font-size: 0.85rem;'>Query ↓ / Key →</td>"
    for w in display_words:
        html += f"<td style='padding: 6px; text-align: center; font-weight: bold; color: #9ca3af; font-size: 0.85rem;'>{w}</td>"
    html += "</tr>"
    
    # Rows
    for i, query_word in enumerate(display_words):
        html += f"<tr><td style='padding: 6px; font-weight: bold; color: #9ca3af; font-size: 0.85rem; white-space: nowrap;'>{query_word}</td>"
        for j in range(n):
            if j <= i:
                # Can attend
                html += f"<td style='padding: 6px; text-align: center; background: #10b981; color: white; border-radius: 4px;'>✓</td>"
            else:
                # Blocked
                html += f"<td style='padding: 6px; text-align: center; background: #dc2626; color: white; opacity: 0.5; border-radius: 4px;'>✗</td>"
        html += "</tr>"
    
    html += "</table>"
    html += "</div>"
    return html


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 👀 Workshop 4 of 6")
    st.markdown("### Attention Mechanism")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="analogy-box">
    <h4>🛸 The Spotlight Analogy</h4>
    <p>Imagine our alien is at a concert with a magical spotlight. 
    Instead of illuminating everyone equally, the spotlight automatically 
    focuses on the most relevant performers!</p>
    <p>When the alien hears "guitar solo," the spotlight brightens 
    on the guitarist. When they hear "drum fill," it shifts to the drummer.</p>
    <p><strong>This is exactly what attention does:</strong> it helps the model 
    focus on the most relevant parts of the input!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔗 Workshop Series
    1. ✅ Tokenization
    2. ✅ Embeddings  
    3. ✅ Vector Databases
    4. 👉 **Attention** ← You are here!
    5. 🔲 Transformers
    6. 🔲 RAG
    """)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown('<p class="main-header">👀 Attention Mechanism Explorer</p>', unsafe_allow_html=True)
st.markdown("### *The magical spotlight that helps models focus on what matters*")

# Data Flow Pipeline
st.markdown("""
<div class="flow-container">
    <h4 style="margin-top: 0; color: #e2e8f0;">📊 Attention Data Flow</h4>
    <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 10px;">
        <div class="flow-step flow-input">📝 Input Tokens</div>
        <span style="color: #64748b; font-size: 1.5rem;">→</span>
        <div class="flow-step flow-process">🔢 Q, K, V Projections</div>
        <span style="color: #64748b; font-size: 1.5rem;">→</span>
        <div class="flow-step flow-process">⚡ Attention Scores</div>
        <span style="color: #64748b; font-size: 1.5rem;">→</span>
        <div class="flow-step flow-process">📊 Softmax Weights</div>
        <span style="color: #64748b; font-size: 1.5rem;">→</span>
        <div class="flow-step flow-output">✨ Context-Aware Output</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Interactive Demo",
    "📊 Attention Types", 
    "🎭 Causal Masking",
    "🧠 Multi-Head Attention"
])


# =============================================================================
# TAB 1: INTERACTIVE DEMO
# =============================================================================

with tab1:
    st.markdown("## 🔬 See Attention in Action")
    st.markdown("Enter a sentence and see how each word attends to others!")
    
    # Example sentences with explanations
    st.markdown("### 💡 Try These Examples")
    example_sentences = {
        "Custom (type your own)": "",
        "🔗 Pronoun Resolution: 'The doctor told the nurse that she was tired'": "The doctor told the nurse that she was tired",
        "🔗 Pronoun Resolution: 'The trophy would not fit in the suitcase because it was too big'": "The trophy would not fit in the suitcase because it was too big",
        "🎯 Subject-Verb: 'The keys to the cabinet are on the table'": "The keys to the cabinet are on the table",
        "🎯 Subject-Verb: 'The dog in the park with the children runs fast'": "The dog in the park with the children runs fast",
        "💭 Semantic: 'The hungry cat ate the fresh fish quickly'": "The hungry cat ate the fresh fish quickly",
        "💭 Semantic: 'The programmer fixed the bug in the code'": "The programmer fixed the bug in the code",
        "🔄 Coreference: 'John gave Mary the book and she thanked him'": "John gave Mary the book and she thanked him",
        "🔄 Coreference: 'The CEO said the company will expand because they have profits'": "The CEO said the company will expand because they have profits",
    }
    
    selected_example = st.selectbox(
        "Choose an example or type your own:",
        list(example_sentences.keys()),
        index=0
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        default_value = example_sentences[selected_example] if example_sentences[selected_example] else "The cat sat on the mat"
        sentence = st.text_input(
            "Enter a sentence:",
            value=default_value,
            help="Try sentences with pronouns like 'it', 'she', 'they' for interesting attention patterns!"
        )
    
    with col2:
        attention_type = st.selectbox(
            "Attention Type:",
            ["scaled", "dot_product", "multi_head"],
            format_func=lambda x: {
                "scaled": "Scaled Dot-Product",
                "dot_product": "Basic Dot-Product",
                "multi_head": "Multi-Head (4 heads)"
            }[x]
        )
    
    # Show pattern hints based on example
    if "Pronoun" in selected_example:
        st.info("👀 **What to look for:** When you select pronouns like 'she', 'it', 'him', see which nouns they attend to most strongly!")
    elif "Subject-Verb" in selected_example:
        st.info("👀 **What to look for:** The verb should attend strongly to its subject, even when other words are in between!")
    elif "Semantic" in selected_example:
        st.info("👀 **What to look for:** Related words (like 'cat'→'ate'→'fish' or 'programmer'→'fixed'→'bug') should show stronger attention!")
    elif "Coreference" in selected_example:
        st.info("👀 **What to look for:** Pronouns 'she', 'him', 'they' should attend back to the people/entities they refer to!")
    
    words = sentence.split()
    
    if len(words) < 2:
        st.warning("Please enter at least 2 words!")
    else:
        # Query selector
        query_idx = st.slider(
            "Select query word (the one doing the 'looking'):",
            0, len(words) - 1, 0
        )
        
        st.markdown(f"### 🔍 Query: **{words[query_idx]}** is attending to...")
        
        # Compute attention
        num_heads = 4 if attention_type == "multi_head" else 1
        output, weights = compute_attention_for_demo(
            words, query_idx, attention_type, num_heads
        )
        
        # Render spotlight visualization
        st.markdown(render_attention_visualization(words, weights, query_idx), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Full heatmap
        st.markdown("### 📊 Full Attention Matrix")
        st.markdown("Each row shows how much that word attends to each column word.")
        st.markdown(render_attention_heatmap(words, weights), unsafe_allow_html=True)
        
        # Metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(words)}</div>
                <div class="metric-label">Sequence Length</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">64</div>
                <div class="metric-label">Embedding Dim</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_heads}</div>
                <div class="metric-label">Attention Heads</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(words) * len(words)}</div>
                <div class="metric-label">Attention Pairs</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: ATTENTION TYPES COMPARISON
# =============================================================================

with tab2:
    st.markdown("## 📊 Comparing Attention Types")
    
    st.markdown("""
    ### Why Different Attention Types?
    
    | Type | Formula | Why Use It? |
    |------|---------|-------------|
    | **Dot-Product** | Q·Kᵀ | Simple, fast, but can have large values |
    | **Scaled Dot-Product** | (Q·Kᵀ) / √d | Prevents softmax saturation |
    | **Multi-Head** | Concat(head₁, ..., headₙ) | Captures different relationship types |
    """)
    
    st.markdown("---")
    
    # Demo sentence with example selector
    st.markdown("### 🔍 Compare on Real Examples")
    
    compare_examples = {
        "Pronoun binding: 'She gave him the book because he asked'": "She gave him the book because he asked",
        "Long-distance dependency: 'The man who the woman saw left quickly'": "The man who the woman saw left quickly",
        "Nested structure: 'I think that she knows that he left'": "I think that she knows that he left",
    }
    
    selected_compare = st.selectbox(
        "Choose an example:",
        list(compare_examples.keys()),
        key="compare_select"
    )
    
    demo_sentence = st.text_input(
        "Compare attention types on:",
        value=compare_examples[selected_compare],
        key="compare_sentence"
    )
    
    demo_words = demo_sentence.split()
    
    if len(demo_words) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Basic Dot-Product")
            _, weights_dot = compute_attention_for_demo(demo_words, 0, "dot_product")
            st.markdown(render_attention_heatmap(demo_words, weights_dot, "Dot-Product Attention"), unsafe_allow_html=True)
            
            st.info("""
            **⚠️ Issue:** Raw dot products can be very large, causing softmax to produce 
            near-one-hot distributions (attention goes almost entirely to one word).
            """)
        
        with col2:
            st.markdown("### Scaled Dot-Product")
            _, weights_scaled = compute_attention_for_demo(demo_words, 0, "scaled")
            st.markdown(render_attention_heatmap(demo_words, weights_scaled, "Scaled Attention"), unsafe_allow_html=True)
            
            st.success("""
            **✅ Solution:** Dividing by √d keeps gradients stable and allows attention 
            to be more evenly distributed when appropriate.
            """)
        
        st.markdown("---")
        
        st.markdown("### 📐 The Scaling Factor Explained")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ```
            Without scaling (d=64):
            dot_product = Q·Kᵀ  
            → values can be ~64× larger!
            → softmax becomes very "peaky"
            
            With scaling:
            scaled = (Q·Kᵀ) / √64 = (Q·Kᵀ) / 8
            → values stay reasonable
            → softmax stays smooth
            ```
            """)
        
        with col2:
            # Visual demonstration
            d = 64
            scale_factor = np.sqrt(d)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">√{d} = {scale_factor:.1f}</div>
                <div class="metric-label">Scaling Factor for dim={d}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: CAUSAL MASKING
# =============================================================================

with tab3:
    st.markdown("## 🎭 Causal Masking (GPT-style)")
    
    st.markdown("""
    ### Why Causal Masking?
    
    When generating text **one word at a time** (like GPT), each word should only 
    "see" the words that came before it. Otherwise, the model would be cheating 
    by looking at the future!
    
    🛸 **Analogy:** Our alien is reading a mystery novel but covers up everything 
    after the current page. They can only use clues from pages they've already read!
    """)
    
    st.markdown("---")
    
    # Better examples for causal masking
    st.markdown("### 🔮 Text Generation Examples")
    
    causal_examples = {
        "Story completion: 'Once upon a time there was a brave knight'": "Once upon a time there was a brave knight",
        "Question answering: 'The capital of France is'": "The capital of France is",
        "Code completion: 'def calculate sum of numbers in'": "def calculate sum of numbers in",
        "Chat response: 'User asked about the weather today so'": "User asked about the weather today so",
    }
    
    selected_causal = st.selectbox(
        "Choose a generation example:",
        list(causal_examples.keys()),
        key="causal_select"
    )
    
    st.info("👀 **What to look for:** Each word can ONLY see words to its left. The last word has the most context!")
    
    causal_sentence = st.text_input(
        "Try causal masking on:",
        value=causal_examples[selected_causal],
        key="causal_sentence"
    )
    
    causal_words = causal_sentence.split()
    
    if len(causal_words) >= 2:
        # Show the mask
        st.markdown(render_causal_mask_visual(causal_words), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Compare bidirectional vs causal
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Bidirectional (BERT-style)")
            _, weights_bi = compute_attention_for_demo(causal_words, 0, "scaled", use_causal=False)
            st.markdown(render_attention_heatmap(causal_words, weights_bi, "Can see ALL words"), unsafe_allow_html=True)
            
            st.info("""
            **Use case:** Understanding text (sentiment analysis, classification)
            - Can see entire sentence
            - Better for comprehension tasks
            """)
        
        with col2:
            st.markdown("### Causal (GPT-style)")
            _, weights_causal = compute_attention_for_demo(causal_words, 0, "scaled", use_causal=True)
            st.markdown(render_attention_heatmap(causal_words, weights_causal, "Can only see past"), unsafe_allow_html=True)
            
            st.success("""
            **Use case:** Generating text (ChatGPT, Claude)
            - Can only see previous words
            - Essential for autoregressive generation
            """)
        
        st.markdown("---")
        
        # Interactive generation simulation
        st.markdown("### 🔮 Simulated Text Generation")
        st.markdown("Watch how the model 'sees' more context as it generates each word:")
        
        for i in range(len(causal_words)):
            visible = causal_words[:i+1]
            hidden = causal_words[i+1:]
            
            visible_html = " ".join([f"<span style='background: #10b981; color: white; padding: 4px 8px; border-radius: 4px; margin: 2px;'>{w}</span>" for w in visible])
            hidden_html = " ".join([f"<span style='background: #374151; color: #6b7280; padding: 4px 8px; border-radius: 4px; margin: 2px; text-decoration: line-through;'>{w}</span>" for w in hidden])
            
            st.markdown(f"""
            <div style='margin: 8px 0;'>
                <strong>Step {i+1}:</strong> {visible_html} {hidden_html}
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 4: MULTI-HEAD ATTENTION
# =============================================================================

with tab4:
    st.markdown("## 🧠 Multi-Head Attention")
    
    st.markdown("""
    ### Why Multiple Heads?
    
    A single attention head can only focus on one type of relationship at a time.
    But language has many types of relationships:
    
    - **Syntactic:** subject-verb agreement ("The cat **sits**")
    - **Semantic:** meaning relationships ("cat" → "furry")  
    - **Positional:** nearby words often relate
    - **Coreference:** "it" → "the ball"
    
    🛸 **Analogy:** Instead of one spotlight, our alien now has a **team of spotlights**, 
    each looking for different patterns! One tracks grammar, another tracks meaning, 
    another tracks pronouns...
    """)
    
    st.markdown("---")
    
    # Multi-head examples with explanations
    st.markdown("### 🎯 Examples Showing Different Head Patterns")
    
    multi_examples = {
        "Pronoun + Action: 'The dog chased the cat and it ran away'": "The dog chased the cat and it ran away",
        "Multiple relationships: 'Mary gave John a book that she bought yesterday'": "Mary gave John a book that she bought yesterday",
        "Complex structure: 'The student who the professor praised received an award'": "The student who the professor praised received an award",
        "Semantic clusters: 'The chef cooked delicious pasta with fresh tomatoes and basil'": "The chef cooked delicious pasta with fresh tomatoes and basil",
    }
    
    selected_multi = st.selectbox(
        "Choose an example:",
        list(multi_examples.keys()),
        key="multi_select"
    )
    
    st.info("""
    👀 **What to look for:** Each head develops a "specialty"!
    - Some heads focus on **nearby words** (positional)
    - Some heads track **pronouns back to nouns** (coreference)  
    - Some heads connect **verbs to subjects** (syntactic)
    - Some heads link **semantically related words** (meaning)
    """)
    
    multi_sentence = st.text_input(
        "Explore multi-head attention on:",
        value=multi_examples[selected_multi],
        key="multi_sentence"
    )
    
    multi_words = multi_sentence.split()
    num_heads = st.slider("Number of attention heads:", 1, 8, 4)
    
    if len(multi_words) >= 2:
        # Use the demo function for consistent meaningful patterns
        output, all_weights = compute_attention_for_demo(
            multi_words, 0, "multi_head", num_heads
        )
        
        # Show each head
        st.markdown("### 👁️ Individual Head Patterns")
        st.markdown("Each head may capture different relationships:")
        
        # Create columns for heads
        cols = st.columns(min(num_heads, 4))
        
        for h in range(num_heads):
            col_idx = h % 4
            with cols[col_idx]:
                st.markdown(f"**Head {h+1}**")
                head_weights = all_weights[h]
                
                # Find the strongest attention for this head
                max_attention = np.unravel_index(
                    np.argmax(head_weights - np.eye(len(multi_words))), 
                    head_weights.shape
                )
                
                if max_attention[0] != max_attention[1]:
                    st.markdown(f"🔍 Focuses on: **{multi_words[max_attention[0]]}** → **{multi_words[max_attention[1]]}**")
                
                st.markdown(render_attention_heatmap(multi_words, head_weights, f"Head {h+1}"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Combined view
        st.markdown("### 🔄 Combined Attention (Average of All Heads)")
        avg_weights = all_weights.mean(axis=0)
        st.markdown(render_attention_heatmap(multi_words, avg_weights, "Combined Multi-Head"), unsafe_allow_html=True)
        
        # Architecture info
        st.markdown("---")
        st.markdown("### 📐 Multi-Head Architecture")
        
        col1, col2, col3 = st.columns(3)
        
        head_dim = 64 // num_heads
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{64}</div>
                <div class="metric-label">Model Dimension</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_heads}</div>
                <div class="metric-label">Number of Heads</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{head_dim}</div>
                <div class="metric-label">Dim per Head</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ```
        Multi-Head Attention Process:
        1. Split input into {num_heads} heads: 64 → {num_heads} × {head_dim}
        2. Each head computes attention independently
        3. Concatenate head outputs: {num_heads} × {head_dim} → 64
        4. Final linear projection
        ```
        """.format(num_heads=num_heads, head_dim=head_dim))


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 20px;">
    <p>🎯 <strong>Workshop 4 of 6:</strong> Attention Mechanism</p>
    <p>Part of the <em>GenAI Self-Build Workshop Series</em></p>
    <p>Next up: 🧠 Workshop 5 - Transformers (Putting it all together!)</p>
</div>
""", unsafe_allow_html=True)
