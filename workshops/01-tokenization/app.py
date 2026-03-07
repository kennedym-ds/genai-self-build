"""
🎯 Workshop 1: Interactive Tokenizer Demo
==========================================

A Streamlit app to explore tokenization strategies interactively.

Run with:
    streamlit run app.py
"""

import streamlit as st
from tokenizer import SimpleTokenizer

# Page config
st.set_page_config(
    page_title="🛸 Tokenizer Explorer",
    page_icon="🛸",
    layout="wide"
)

# Custom CSS for token visualization
st.markdown("""
<style>
    .token-box {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 14px;
    }
    .token-char { background-color: #3b82f6; color: white; }
    .token-word { background-color: #8b5cf6; color: white; }
    .token-bpe { background-color: #10b981; color: white; }
    .token-id { background-color: #1f2937; color: #9ca3af; font-size: 11px; }
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
</style>
""", unsafe_allow_html=True)

# Header
st.title("🛸 Tokenizer Explorer")
st.markdown("### *How does an alien learn to read English?*")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["🔬 Interactive Demo", "📊 How It Works", "🆚 Compare Strategies"])

# Alien analogy in sidebar
with st.sidebar:
    st.markdown("## 🛸 The Alien Analogy")
    st.markdown("""
    Imagine an alien lands on Earth and needs to learn English. 
    They don't know what "words" are—just symbols!
    
    But the alien notices **patterns**:
    - `"the"` appears everywhere → **θ**
    - `"ing"` ends many words → **ω**
    - `"tion"` is common → **τ**
    
    After studying millions of texts, the alien builds a **codebook**.
    
    **This is exactly what tokenization does!**
    """)
    
    st.divider()
    
    st.markdown("## ⚙️ Settings")
    
    # Strategy selection
    strategy = st.radio(
        "Tokenization Strategy",
        ["char", "word", "bpe"],
        format_func=lambda x: {
            "char": "📝 Character (spell it out)",
            "word": "📖 Word (dictionary pages)", 
            "bpe": "📦 BPE (smart grouping)"
        }[x],
        help="Switch strategies to see different tokenization approaches"
    )
    
    # Vocab size for word/bpe
    if strategy == "char":
        vocab_size = 100  # Not used for char
        st.info("📝 **Character** mode uses all unique characters as vocabulary - no size limit needed!")
    else:
        vocab_size = st.slider(
            "Vocabulary Size", 
            min_value=30, 
            max_value=200, 
            value=100,
            help="Smaller = more tokens per text, Larger = fewer tokens but bigger codebook"
        )
        st.caption(f"🎯 Training with **{vocab_size}** vocab limit")
    
    # Clear cache button to force re-training
    st.divider()
    if st.button("🔄 Retrain Tokenizer", use_container_width=True, help="Force the tokenizer to retrain with current settings"):
        st.cache_resource.clear()
        st.rerun()
    
    st.divider()
    st.markdown("### 🎯 Workshop 1 of 6")
    st.markdown("*GenAI Self-Build Series*")

# Default corpus - larger to show BPE benefits
DEFAULT_CORPUS = """The quick brown fox jumps over the lazy dog.
Machine learning is transforming how we build software.
Python is a popular programming language for programming tasks.
Tokenization is the first step in natural language processing.
The fox is quick and clever and jumps quickly.
Learning to code is fun and rewarding.
Deep learning and machine learning are transforming technology.
Natural language processing enables machines to understand text.
The programmer is programming a program for processing data.
Transformers are transforming how we transform information.
Running runners run while swimming swimmers swim.
The teacher is teaching students who are learning.
Understanding understanding helps with misunderstanding.
Happiness and unhappiness depend on our thinking and rethinking."""

# Initialize or update tokenizer
@st.cache_resource
def get_tokenizer(strategy, corpus_hash, vocab_size):
    tokenizer = SimpleTokenizer(strategy=strategy)
    corpus = [line.strip() for line in corpus_hash.split('\n') if line.strip()]
    tokenizer.train(corpus, vocab_size=vocab_size)
    return tokenizer

# ============================================================================
# TAB 1: Interactive Demo
# ============================================================================
with tab1:
    # Data flow visualization at top
    st.markdown("## 🔄 The Tokenization Pipeline")
    
    flow_cols = st.columns([2, 1, 2, 1, 2, 1, 2])
    
    with flow_cols[0]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">📝</div>
            <div><strong>Text</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Human readable</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[1]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[2]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">🛸</div>
            <div><strong>Tokenizer</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">The "alien" codebook</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[3]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[4]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">🔢</div>
            <div><strong>Token IDs</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Numbers for AI</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[5]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[6]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">🧠</div>
            <div><strong>Neural Net</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">LLM processing</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.divider()
    
    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("## 📚 Training Corpus")
        st.caption("This is what the alien studies to learn patterns")
        
        corpus_text = st.text_area(
            "Training text (one sentence per line):",
            value=DEFAULT_CORPUS,
            height=200,
            label_visibility="collapsed"
        )
        
        corpus = [line.strip() for line in corpus_text.split('\n') if line.strip()]

    with col2:
        st.markdown("## ✏️ Test Input")
        st.caption("Type something for the alien to read")
        
        test_text = st.text_input(
            "Text to tokenize:",
            value="The quick fox is learning.",
            label_visibility="collapsed"
        )
        
        # Show what strategy does
        strategy_info = {
            "char": ("📝 Character", "Each character becomes a token. Simple but creates long sequences.", "#3b82f6"),
            "word": ("📖 Word", "Each word becomes a token. Short sequences but huge vocabulary.", "#8b5cf6"),
            "bpe": ("📦 BPE", "Common character pairs merge into tokens. Best of both worlds!", "#10b981")
        }
        name, desc, color = strategy_info[strategy]
        st.markdown(f"""
        <div class="info-card" style="border-left-color: {color};">
            <strong>{name}</strong><br/>
            {desc}
        </div>
        """, unsafe_allow_html=True)

    # Always create tokenizer (cached)
    corpus_hash = corpus_text  # Use as cache key
    tokenizer = get_tokenizer(strategy, corpus_hash, vocab_size)
    
    # Show current settings prominently
    settings_cols = st.columns(3)
    with settings_cols[0]:
        st.success(f"**Strategy:** {strategy.upper()}")
    with settings_cols[1]:
        st.info(f"**Vocab Size:** {tokenizer.vocab_size()} tokens")
    with settings_cols[2]:
        if strategy == "bpe":
            st.warning(f"**Merges Learned:** {len(tokenizer.merges)}")
        else:
            st.warning(f"**Corpus Lines:** {len(corpus)}")

    st.divider()

    # Results
    st.markdown("## 🔍 Tokenization Results")

    # Encode the text
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    # Get individual tokens for display
    if strategy == "char":
        tokens = [tokenizer.inverse_vocab.get(id, "?") for id in encoded]
    elif strategy == "word":
        tokens = [tokenizer.inverse_vocab.get(id, "<UNK>") for id in encoded]
    else:  # bpe
        tokens = []
        for id in encoded:
            token = tokenizer.inverse_vocab.get(id, "?")
            if token != "</w>":
                tokens.append(token.replace("</w>", ""))

    # Create three columns for comparison
    result_cols = st.columns(3)

    with result_cols[0]:
        st.markdown("### 📥 Input")
        st.code(test_text, language=None)
        st.metric("Characters", len(test_text))

    with result_cols[1]:
        st.markdown("### 🔢 Token IDs")
        st.code(str(encoded), language=None)
        st.metric("Tokens", len(encoded))

    with result_cols[2]:
        st.markdown("### 📤 Decoded")
        st.code(decoded, language=None)
        match = "✅ Match!" if decoded.lower().replace(".", "") == test_text.lower().replace(".", "").replace(",", "") else "⚠️ Normalized"
        st.metric("Roundtrip", match)

    # Visual token display
    st.markdown("### 🎨 Token Visualization")
    st.caption("Each colored box is one token sent to the neural network")

    token_html = ""
    for i, (token, id) in enumerate(zip(tokens, encoded)):
        display_token = token if token.strip() else "␣"
        if strategy == "char":
            display_token = token if token != " " else "␣"
        token_html += f'''
        <span class="token-box token-{strategy}">
            {display_token}
            <span class="token-id">#{id}</span>
        </span>
        '''

    st.markdown(f'<div style="line-height: 2.5;">{token_html}</div>', unsafe_allow_html=True)

    # Vocabulary explorer
    with st.expander("📖 Vocabulary Explorer", expanded=False):
        vocab_cols = st.columns(2)
        
        with vocab_cols[0]:
            st.markdown(f"**Vocabulary Size:** {tokenizer.vocab_size()}")
            
            # Show sample of vocabulary
            st.markdown("**Sample tokens:**")
            sample_vocab = dict(list(tokenizer.vocab.items())[:20])
            st.json(sample_vocab)
        
        with vocab_cols[1]:
            if strategy == "bpe" and tokenizer.merges:
                st.markdown(f"**Merge Rules Learned:** {len(tokenizer.merges)}")
                st.markdown("**First 10 merges:**")
                for i, merge in enumerate(tokenizer.merges[:10]):
                    st.code(f"{merge[0]} + {merge[1]} → {merge[0]}{merge[1]}", language=None)

# ============================================================================
# TAB 2: How It Works
# ============================================================================
with tab2:
    st.markdown("## 📊 How Tokenization Works")
    
    # Visual explanation of each strategy
    st.markdown("### 🛸 The Alien's Three Methods")
    
    method_cols = st.columns(3)
    
    with method_cols[0]:
        st.markdown("""
        ### 📝 Character Level
        **Analogy:** Spelling on the phone
        
        ```
        "Hello" 
           ↓
        ['H','e','l','l','o']
           ↓
        [7, 4, 11, 11, 14]
        ```
        
        **Pros:**
        - ✅ Tiny vocabulary (~100)
        - ✅ Handles any text
        
        **Cons:**
        - ❌ Very long sequences
        - ❌ Loses word meaning
        """)
    
    with method_cols[1]:
        st.markdown("""
        ### 📖 Word Level
        **Analogy:** Dictionary page numbers
        
        ```
        "Hello World"
           ↓
        ['hello', 'world']
           ↓
        [4821, 9023]
        ```
        
        **Pros:**
        - ✅ Short sequences
        - ✅ Preserves meaning
        
        **Cons:**
        - ❌ Huge vocabulary (100K+)
        - ❌ Unknown words fail!
        """)
    
    with method_cols[2]:
        st.markdown("""
        ### 📦 BPE (Subword)
        **Analogy:** Shipping boxes
        
        ```
        "unhappiness"
           ↓
        ['un', 'happi', 'ness']
           ↓
        [42, 891, 127]
        ```
        
        **Pros:**
        - ✅ Balanced sequences
        - ✅ Handles new words
        
        **This is what GPT uses!** 🎯
        """)
    
    st.divider()
    
    # BPE Algorithm visualization
    st.markdown("### 🔄 BPE Algorithm: How the Alien Learns")
    
    st.markdown("""
    The alien learns by watching which letter pairs appear together most often:
    """)
    
    bpe_steps = st.columns(5)
    
    with bpe_steps[0]:
        st.markdown("""
        **Step 1: Start**
        ```
        l o w e r
        l o w
        ```
        All individual chars
        """)
    
    with bpe_steps[1]:
        st.markdown("""
        **Step 2: Count**
        ```
        lo: 2 times ⭐
        ow: 2 times
        er: 1 time
        ```
        Find most common
        """)
    
    with bpe_steps[2]:
        st.markdown("""
        **Step 3: Merge**
        ```
        lo w e r
        lo w
        ```
        `l` + `o` → `lo`
        """)
    
    with bpe_steps[3]:
        st.markdown("""
        **Step 4: Repeat**
        ```
        low e r
        low
        ```
        `lo` + `w` → `low`
        """)
    
    with bpe_steps[4]:
        st.markdown("""
        **Step 5: Done!**
        ```
        low er
        low
        ```
        Common subwords
        """)
    
    st.divider()
    
    # Why this matters
    st.markdown("### 🧠 Why This Matters for LLMs")
    
    llm_cols = st.columns(2)
    
    with llm_cols[0]:
        st.markdown("""
        #### The OOV Problem
        
        Word tokenizers **break** on unknown words:
        
        ```python
        vocab = {"hello": 1, "world": 2}
        
        encode("TensorFlow")  
        # → ??? Not in vocab!
        ```
        
        BPE solves this:
        ```python
        encode("TensorFlow")
        # → ["Tensor", "Flow"]
        # Known subwords! ✅
        ```
        """)
    
    with llm_cols[1]:
        st.markdown("""
        #### Real-World Tokenizers
        
        | Model | Tokenizer | Vocab Size |
        |-------|-----------|------------|
        | GPT-4 | tiktoken (BPE) | ~100K |
        | Claude | BPE variant | ~100K |
        | BERT | WordPiece | ~30K |
        | LLaMA | SentencePiece | ~32K |
        
        **All use subword tokenization!**
        """)

# ============================================================================
# TAB 3: Compare Strategies
# ============================================================================
with tab3:
    st.markdown("## 🆚 Strategy Comparison")
    
    # The key insight: unknown words!
    st.markdown("""    
    ### 🎯 The Real Test: Unknown Words
    
    The **killer feature** of BPE isn't compression—it's handling words the tokenizer has **never seen before**!
    """)
    
    # Preset examples that show the difference
    example_options = {
        "Known words (all work)": "The quick fox is learning.",
        "🔥 Unknown word test": "The TensorFlow programmer is relearning.",
        "🔥 Made-up word": "The unhappifying chatbot was reprogramming itself.",
        "🔥 Technical jargon": "Transformerification enables better tokenization.",
        "Custom input": ""
    }
    
    selected_example = st.radio(
        "Choose a test case:",
        list(example_options.keys()),
        horizontal=True,
        help="Try the 🔥 examples to see BPE's advantage!"
    )
    
    if selected_example == "Custom input":
        compare_text = st.text_input(
            "Enter your own text:",
            value="Try words like unhappiness or reprogramming",
            key="compare_input"
        )
    else:
        compare_text = example_options[selected_example]
        st.code(compare_text, language=None)
    
    # Get all three tokenizers
    char_tok = get_tokenizer("char", corpus_text, 100)
    word_tok = get_tokenizer("word", corpus_text, vocab_size)
    bpe_tok = get_tokenizer("bpe", corpus_text, vocab_size)
    
    # Encode with each
    char_enc = char_tok.encode(compare_text)
    word_enc = word_tok.encode(compare_text)
    bpe_enc = bpe_tok.encode(compare_text)
    
    # Check for unknown words
    words = [w.lower() for w in compare_text.replace('.', '').replace(',', '').split()]
    unknown_words = [w for w in words if w not in word_tok.vocab]
    
    if unknown_words:
        st.error(f"⚠️ **Unknown words detected:** {', '.join(unknown_words)}")
        st.markdown("Watch how each strategy handles these!")
    
    # Comparison table
    st.markdown("### 📊 Results Comparison")
    
    compare_cols = st.columns(3)
    
    with compare_cols[0]:
        st.markdown("#### 📝 Character")
        st.metric("Tokens", len(char_enc))
        st.metric("Vocab Size", char_tok.vocab_size())
        st.caption("Every character = 1 token")
        
        # Mini visualization
        char_tokens = [char_tok.inverse_vocab.get(id, "?") for id in char_enc[:15]]
        char_html = "".join([f'<span class="token-box token-char">{t if t != " " else "␣"}</span>' for t in char_tokens])
        st.markdown(f'{char_html}{"..." if len(char_enc) > 15 else ""}', unsafe_allow_html=True)
    
    with compare_cols[1]:
        st.markdown("#### 📖 Word")
        st.metric("Tokens", len(word_enc))
        st.metric("Vocab Size", word_tok.vocab_size())
        
        word_tokens = [word_tok.inverse_vocab.get(id, "<UNK>") for id in word_enc]
        unk_count = word_tokens.count("<UNK>")
        if unk_count > 0:
            st.caption(f"❌ {unk_count} unknown word(s)!")
        else:
            st.caption("✅ All words known")
        
        # Highlight UNK tokens in red
        word_html = ""
        for t in word_tokens:
            if t == "<UNK>":
                word_html += f'<span class="token-box" style="background-color: #dc2626; color: white;">{t}</span>'
            else:
                word_html += f'<span class="token-box token-word">{t}</span>'
        st.markdown(word_html, unsafe_allow_html=True)
    
    with compare_cols[2]:
        st.markdown("#### 📦 BPE")
        st.metric("Tokens", len(bpe_enc))
        st.metric("Vocab Size", bpe_tok.vocab_size())
        st.caption("✅ Breaks into known subwords!")
        
        bpe_tokens = []
        for id in bpe_enc:
            token = bpe_tok.inverse_vocab.get(id, "?")
            if token != "</w>":
                bpe_tokens.append(token.replace("</w>", ""))
        bpe_html = "".join([f'<span class="token-box token-bpe">{t}</span>' for t in bpe_tokens])
        st.markdown(bpe_html, unsafe_allow_html=True)
        
        # Show that BPE handled unknown words
        if unknown_words:
            st.success("🎉 No <UNK> tokens! BPE broke unknown words into known pieces.")
    
    st.divider()
    
    # Efficiency chart - but make it clear Word's "compression" is fake
    st.markdown("### 📈 Efficiency Analysis")
    
    import pandas as pd
    
    # Count UNK tokens for word
    word_tokens = [word_tok.inverse_vocab.get(id, "<UNK>") for id in word_enc]
    unk_count = word_tokens.count("<UNK>")
    
    # Build the table with quality indicator
    efficiency_data = pd.DataFrame({
        "Strategy": ["📝 Character", "📖 Word", "📦 BPE"],
        "Tokens": [len(char_enc), len(word_enc), len(bpe_enc)],
        "Unknown": ["0 ✅", f"{unk_count} {'❌' if unk_count > 0 else '✅'}", "0 ✅"],
        "Quality": [
            "✅ Lossless",
            "❌ LOSSY!" if unk_count > 0 else "✅ Lossless",
            "✅ Lossless"
        ],
        "Verdict": [
            "Too many tokens",
            "⚠️ CHEATING! Lost info" if unk_count > 0 else "Good",
            "🏆 Best balance!"
        ]
    })
    
    st.dataframe(efficiency_data, use_container_width=True, hide_index=True)
    
    # Explain the "cheating"
    if unk_count > 0:
        st.error(f"""
        ⚠️ **Word tokenizer is "cheating"!**  
        
        It shows fewer tokens ({len(word_enc)}) but **{unk_count} are `<UNK>`** - meaning it gave up and lost information!  
        The LLM would have no idea what those words meant.
        
        BPE uses more tokens ({len(bpe_enc)}) but **preserves all information** by breaking unknown words into known pieces.
        """)
    
    # Bar chart with better labels
    st.markdown("### 📊 Token Count Comparison")
    chart_data = pd.DataFrame({
        "Strategy": ["📝 Char", "📖 Word", "📦 BPE"],
        "Token Count": [len(char_enc), len(word_enc), len(bpe_enc)]
    })
    st.bar_chart(chart_data.set_index("Strategy"))
    
    # Key insight callout
    st.divider()
    st.markdown("### 💡 Key Insight")
    
    insight_cols = st.columns(2)
    with insight_cols[0]:
        st.markdown("""
        **Why BPE wins:**
        
        | Scenario | Word | BPE |
        |----------|------|-----|
        | Known words | ✅ Great | ✅ Great |
        | Unknown words | ❌ `<UNK>` | ✅ Subwords |
        | New slang | ❌ Fails | ✅ Works |
        | Typos | ❌ Fails | ✅ Works |
        """)
    
    with insight_cols[1]:
        st.markdown("""
        **Real-world example:**
        
        Word tokenizer on "ChatGPT":
        ```
        → <UNK>  (never seen it!)
        ```
        
        BPE tokenizer on "ChatGPT":
        ```
        → ["Chat", "G", "PT"]  (known pieces!)
        ```
        """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🛸 Workshop 1: Tokenization | GenAI Self-Build Series</p>
    <p><em>Remember: The alien (algorithm) learns patterns to build a codebook (vocabulary)!</em></p>
</div>
""", unsafe_allow_html=True)
