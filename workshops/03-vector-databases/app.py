"""
🎯 Workshop 3: Interactive Vector Database Demo
================================================

A Streamlit app to explore vector databases interactively.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import time
from vector_db import SimpleVectorDB

# Page config
st.set_page_config(
    page_title="📚 Vector DB Explorer",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .vector-box {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 12px;
    }
    .strategy-flat { background-color: #3b82f6; color: white; }
    .strategy-lsh { background-color: #8b5cf6; color: white; }
    .strategy-ivf { background-color: #10b981; color: white; }
    .info-card {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 0 8px 8px 0;
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
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .cluster-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Check if qdrant is available
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Header
st.title("📚 Vector Database Explorer")
st.markdown("### *The alien's library with magic shelves*")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Interactive Demo", "📊 How It Works", "🆚 Compare Strategies", "🚀 Production: Qdrant"])

# Sidebar
with st.sidebar:
    st.markdown("## 📚 The Library Analogy")
    st.markdown("""
    Our alien has a MAP of word meanings (embeddings).
    But with MILLIONS of words, checking each one is too slow!
    
    The alien builds a **MAGIC LIBRARY**:
    - 📖 Similar books on nearby shelves
    - 🏃 Walk to the right section first
    - ✨ Find what you need FAST!
    
    **This is what vector databases do!**
    """)
    
    st.divider()
    
    st.markdown("## ⚙️ Settings")
    
    # Database size
    num_vectors = st.slider(
        "Number of vectors",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="How many vectors to store"
    )
    
    # Dimensions
    dimensions = st.slider(
        "Vector dimensions",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Size of each vector"
    )
    
    # Strategy
    strategy = st.radio(
        "Search Strategy",
        ["flat", "lsh", "ivf"],
        format_func=lambda x: {
            "flat": "📖 Flat (Brute Force)",
            "lsh": "🎲 LSH (Hashing)",
            "ivf": "📊 IVF (Clustering)"
        }[x]
    )
    
    # Strategy-specific settings
    if strategy == 'lsh':
        num_tables = st.slider("Hash tables", 5, 20, 10)
        num_bits = st.slider("Bits per hash", 4, 16, 8)
    elif strategy == 'ivf':
        num_clusters = st.slider("Clusters", 5, 50, 20)
        nprobe = st.slider("Clusters to search", 1, 10, 3)
    
    st.divider()
    st.markdown("### 🎯 Workshop 3 of 6")
    st.markdown("*GenAI Self-Build Series*")

# Create sample data with clusters
@st.cache_data
def create_sample_data(num_vectors, dimensions, num_clusters=5):
    """Create clustered sample data to simulate real embeddings."""
    np.random.seed(42)
    
    vectors_per_cluster = num_vectors // num_clusters
    all_vectors = []
    all_labels = []
    
    categories = ["🔧 Technology", "🧬 Science", "🎨 Arts", "⚽ Sports", "🍳 Food"]
    
    for i in range(num_clusters):
        # Create cluster with some spread
        cluster = np.random.randn(vectors_per_cluster, dimensions) * 0.3
        # Shift cluster in a unique direction
        cluster[:, i % dimensions] += 2
        all_vectors.append(cluster)
        all_labels.extend([categories[i % len(categories)]] * vectors_per_cluster)
    
    # Handle remaining vectors
    remaining = num_vectors - len(all_labels)
    if remaining > 0:
        extra = np.random.randn(remaining, dimensions) * 0.3
        all_vectors.append(extra)
        all_labels.extend(["📝 Other"] * remaining)
    
    vectors = np.vstack(all_vectors)
    # Normalize
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    
    return vectors, all_labels

# Initialize database
@st.cache_resource
def get_database(_strategy, _num_vectors, _dimensions, **kwargs):
    """Create and populate database."""
    vectors, labels = create_sample_data(_num_vectors, _dimensions)
    
    db = SimpleVectorDB(strategy=_strategy, dimensions=_dimensions, **kwargs)
    
    ids = [f"doc_{i}" for i in range(_num_vectors)]
    db.add_batch(ids, vectors)
    db.build_index()
    
    return db, vectors, labels, ids

# ============================================================================
# TAB 1: Interactive Demo
# ============================================================================
with tab1:
    # Data flow visualization
    st.markdown("## 🔄 The Vector Search Pipeline")
    
    flow_cols = st.columns([2, 1, 2, 1, 2, 1, 2])
    
    with flow_cols[0]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">📄</div>
            <div><strong>Documents</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Stored as vectors</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[1]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[2]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">📚</div>
            <div><strong>Index</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Organized for speed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[3]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[4]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">🔍</div>
            <div><strong>Query</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Find similar</div>
        </div>
        """, unsafe_allow_html=True)
    
    with flow_cols[5]:
        st.markdown('<div class="flow-arrow" style="text-align:center; padding-top:20px;">→</div>', unsafe_allow_html=True)
    
    with flow_cols[6]:
        st.markdown("""
        <div class="flow-step">
            <div style="font-size: 28px;">🎯</div>
            <div><strong>Results</strong></div>
            <div style="font-size: 11px; color: #9ca3af;">Top-k matches</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Get database based on settings
    if strategy == 'lsh':
        db, vectors, labels, ids = get_database(
            strategy, num_vectors, dimensions,
            num_tables=num_tables, num_bits=num_bits
        )
    elif strategy == 'ivf':
        db, vectors, labels, ids = get_database(
            strategy, num_vectors, dimensions,
            num_clusters=num_clusters, nprobe=nprobe
        )
    else:
        db, vectors, labels, ids = get_database(
            strategy, num_vectors, dimensions
        )
    
    # Stats display
    st.markdown("## 📊 Database Statistics")
    
    stats_cols = st.columns(4)
    with stats_cols[0]:
        st.metric("📄 Vectors", f"{db.size():,}")
    with stats_cols[1]:
        st.metric("📐 Dimensions", dimensions)
    with stats_cols[2]:
        st.metric("📈 Strategy", strategy.upper())
    with stats_cols[3]:
        stats = db.get_stats()
        if strategy == 'lsh':
            st.metric("🎲 Hash Tables", stats.get('num_tables', 'N/A'))
        elif strategy == 'ivf':
            st.metric("📊 Clusters", stats.get('num_clusters', 'N/A'))
        else:
            st.metric("📖 Index", "Flat")
    
    st.divider()
    
    # Search interface
    st.markdown("## 🔍 Search Demo")
    
    search_cols = st.columns([1, 2])
    
    with search_cols[0]:
        # Select a query document
        query_idx = st.selectbox(
            "Select a document to query:",
            range(min(100, num_vectors)),
            format_func=lambda i: f"doc_{i} ({labels[i]})"
        )
        
        top_k = st.slider("Results to return (top_k)", 1, 20, 10)
        
        if st.button("🔍 Search!", type="primary", use_container_width=True):
            st.session_state.do_search = True
        else:
            if 'do_search' not in st.session_state:
                st.session_state.do_search = True
    
    with search_cols[1]:
        if st.session_state.get('do_search', False):
            query = vectors[query_idx]
            query_label = labels[query_idx]
            
            # Time the search
            start_time = time.time()
            results = db.search(query, top_k=top_k)
            search_time = (time.time() - start_time) * 1000
            
            st.markdown(f"### 🎯 Results for doc_{query_idx} ({query_label})")
            st.caption(f"⏱️ Search time: {search_time:.3f}ms")
            
            # Display results
            for i, (doc_id, score) in enumerate(results):
                idx = int(doc_id.split('_')[1])
                result_label = labels[idx]
                
                # Check if same category
                match = "✅" if result_label == query_label else "❌"
                
                # Create progress bar
                bar_width = int(max(0, score) * 100)
                
                st.markdown(f"""
                <div style="margin: 5px 0; padding: 8px; background-color: #1e293b; border-radius: 5px;">
                    <span style="font-weight: bold;">{i+1}. {doc_id}</span>
                    <span style="margin-left: 10px;">{result_label}</span>
                    <span style="float: right;">{match} {score:.4f}</span>
                    <div style="margin-top: 5px; width: 100%; height: 8px; background: #374151; border-radius: 4px;">
                        <div style="width: {bar_width}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6); border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Accuracy metric
            same_category = sum(1 for doc_id, _ in results 
                               if labels[int(doc_id.split('_')[1])] == query_label)
            st.metric("Category Accuracy", f"{same_category}/{len(results)}")
    
    st.divider()
    
    # 2D Visualization
    st.markdown("## 🎨 Vector Space Visualization")
    st.caption("Vectors projected to 2D - colors show categories, similar items cluster together")
    
    # Simple 2D projection
    from numpy.linalg import svd
    
    # Sample vectors for visualization (limit for performance)
    sample_size = min(500, len(vectors))
    np.random.seed(42)  # Fixed seed for consistent sampling
    sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
    sample_vectors = vectors[sample_indices]
    sample_labels = [labels[i] for i in sample_indices]
    
    # PCA projection
    centered = sample_vectors - sample_vectors.mean(axis=0)
    try:
        U, S, Vt = svd(centered, full_matrices=False)
        coords_2d = U[:, :2] * S[:2]
        
        # Normalize
        coords_2d[:, 0] = (coords_2d[:, 0] - coords_2d[:, 0].min()) / (coords_2d[:, 0].max() - coords_2d[:, 0].min() + 1e-10)
        coords_2d[:, 1] = (coords_2d[:, 1] - coords_2d[:, 1].min()) / (coords_2d[:, 1].max() - coords_2d[:, 1].min() + 1e-10)
        
        import pandas as pd
        
        chart_data = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'category': sample_labels
        })
        
        st.scatter_chart(
            chart_data,
            x='x',
            y='y',
            color='category',
            size=20,
            height=400
        )
    except Exception as e:
        st.warning(f"Could not create visualization: {e}")


# ============================================================================
# TAB 2: How It Works
# ============================================================================
with tab2:
    st.markdown("## 📊 How Vector Databases Work")
    
    # The problem
    st.markdown("### 🤔 The Problem: Finding Similar Vectors")
    
    st.markdown("""
    Imagine you have **1 million documents** as vectors (from Workshop 2).
    To find similar documents, you need to compare your query to ALL of them!
    
    | Documents | Comparisons | Time @ 1M ops/sec |
    |-----------|-------------|-------------------|
    | 1,000 | 1,000 | 0.001 seconds |
    | 100,000 | 100,000 | 0.1 seconds |
    | 1,000,000 | 1,000,000 | 1 second |
    | 100,000,000 | 100,000,000 | **100 seconds!** |
    
    **We need a smarter approach!**
    """)
    
    st.divider()
    
    # Three strategies
    st.markdown("### 📚 Three Indexing Strategies")
    
    strat_cols = st.columns(3)
    
    with strat_cols[0]:
        st.markdown("""
        ### 📖 Flat (Brute Force)
        
        **Analogy:** Checking every book in the library
        
        ```
        for each vector in database:
            compute similarity(query, vector)
        return top_k highest
        ```
        
        **Pros:**
        - ✅ 100% accurate (exact search)
        - ✅ Simple to implement
        - ✅ No index needed
        
        **Cons:**
        - ❌ O(n) - checks EVERY vector
        - ❌ Too slow for large databases
        
        **Use when:** < 10,000 vectors
        """)
    
    with strat_cols[1]:
        st.markdown("""
        ### 🎲 LSH (Locality Sensitive Hashing)
        
        **Analogy:** Magic coins that land same way for similar items
        
        ```
        hash(query) → bucket_id
        candidates = bucket[bucket_id]
        rank candidates by similarity
        ```
        
        **Pros:**
        - ✅ Sub-linear time O(1) to O(n)
        - ✅ Good for very high dimensions
        - ✅ Memory efficient
        
        **Cons:**
        - ❌ Approximate (may miss some)
        - ❌ Tuning required
        
        **Use when:** Very large, high-dim data
        """)
    
    with strat_cols[2]:
        st.markdown("""
        ### 📊 IVF (Inverted File Index)
        
        **Analogy:** Library organized into sections
        
        ```
        cluster vectors with k-means
        at query time:
            find nearest clusters
            search only those clusters
        ```
        
        **Pros:**
        - ✅ Good accuracy/speed balance
        - ✅ Tunable via nprobe
        - ✅ Works well in practice
        
        **Cons:**
        - ❌ Needs training (k-means)
        - ❌ Still approximate
        
        **Use when:** Medium to large datasets
        """)
    
    st.divider()
    
    # Visual explanation
    st.markdown("### 🎯 Visual: How Each Strategy Searches")
    
    viz_cols = st.columns(3)
    
    with viz_cols[0]:
        st.markdown("""
        **Flat: Check Everything**
        ```
        Query: ★
        
        ● ● ● ● ●
        ● ● ● ● ●
        ● ★ ● ● ●  ← Check ALL
        ● ● ● ● ●
        ● ● ● ● ●
        
        Checks: 25 vectors
        ```
        """)
    
    with viz_cols[1]:
        st.markdown("""
        **LSH: Hash to Buckets**
        ```
        Query: ★ → Hash: "1011"
        
        Bucket "1011":
        [●, ●, ★, ●]  ← Only check bucket!
        
        Other buckets ignored
        
        Checks: 4 vectors
        ```
        """)
    
    with viz_cols[2]:
        st.markdown("""
        **IVF: Search Near Clusters**
        ```
        Query: ★
        
        Cluster A: [●●●]
        Cluster B: [●●★●] ← Nearest!
        Cluster C: [●●●●●]
        
        Only search Cluster B
        
        Checks: 4 vectors
        ```
        """)
    
    st.divider()
    
    # Real world
    st.markdown("### 🌍 Real-World Vector Databases")
    
    st.markdown("""
    | Database | Algorithms | Used By |
    |----------|------------|---------|
    | **Pinecone** | IVF, HNSW | OpenAI plugins |
    | **Weaviate** | HNSW | Hybrid search apps |
    | **Qdrant** | HNSW | Recommendation systems |
    | **Milvus** | IVF, HNSW, more | Enterprise AI |
    | **FAISS** (library) | All of the above | Meta, research |
    | **Chroma** | HNSW | LangChain, local AI |
    
    Most use **HNSW** (Hierarchical Navigable Small Worlds) - a graph-based approach
    that's even faster than IVF for most use cases!
    """)


# ============================================================================
# TAB 3: Compare Strategies
# ============================================================================
with tab3:
    st.markdown("## 🆚 Strategy Comparison")
    
    st.markdown("""
    ### 🎯 The Key Insight: Scale Matters!
    
    At small scale (< 10K vectors), **Flat search is fine** - checking everything is fast enough.
    The magic of approximate methods only shows at **larger scales**.
    
    Below we show two views:
    1. **Theoretical scaling** - how methods behave as data grows
    2. **Live benchmark** - what you can see on your machine
    """)
    
    st.divider()
    
    # ===== SECTION 1: THEORETICAL SCALING =====
    st.markdown("### 📈 How Search Time Scales (Theoretical)")
    
    st.markdown("""
    This chart shows **relative search time** as database size grows.
    Notice how Flat grows linearly while IVF stays nearly constant!
    """)
    
    import pandas as pd
    
    # Create theoretical scaling data
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    size_labels = ["1K", "10K", "100K", "1M", "10M"]
    
    # Theoretical scaling (normalized)
    flat_time = [s / 1000 for s in sizes]  # O(n) - linear
    ivf_time = [np.sqrt(s) / 30 for s in sizes]  # O(sqrt(n)) approx
    lsh_time = [1 + np.log10(s) for s in sizes]  # O(log n) approx
    
    scaling_df = pd.DataFrame({
        'Database Size': size_labels * 3,
        'Relative Search Time': flat_time + ivf_time + lsh_time,
        'Strategy': ['📖 Flat'] * 5 + ['📊 IVF'] * 5 + ['🎲 LSH'] * 5
    })
    
    # Simple bar chart comparison at different scales
    scale_cols = st.columns(5)
    for i, (size, label) in enumerate(zip(sizes, size_labels)):
        with scale_cols[i]:
            flat_t = size / 1000
            ivf_t = np.sqrt(size) / 30
            
            st.markdown(f"**{label} vectors**")
            st.markdown(f"📖 Flat: {flat_t:.0f}x")
            st.markdown(f"📊 IVF: {ivf_t:.1f}x")
            
            if flat_t > 10:
                speedup = flat_t / ivf_t
                st.markdown(f"🚀 **{speedup:.0f}x faster!**")
    
    st.info("""
    💡 **Why approximate methods win at scale:**
    - At 1K vectors: Flat checks 1,000 items, IVF checks ~100 → Similar speed
    - At 1M vectors: Flat checks 1,000,000 items, IVF checks ~1,000 → **1000x faster!**
    """)
    
    st.divider()
    
    # ===== SECTION 2: IVF NPROBE DEMO =====
    st.markdown("### 🎚️ IVF Trade-off: nprobe Parameter")
    
    st.markdown("""
    IVF organizes vectors into clusters. The `nprobe` parameter controls how many clusters to search:
    - **Low nprobe** = Fast but may miss some results
    - **High nprobe** = Slower but finds more results
    """)
    
    nprobe_demo_cols = st.columns([1, 2])
    
    with nprobe_demo_cols[0]:
        demo_vectors = 5000
        demo_clusters = 50
        demo_nprobe = st.slider("nprobe (clusters to search)", 1, 20, 5, key="nprobe_demo")
        
        # Calculate theoretical metrics
        vectors_per_cluster = demo_vectors // demo_clusters
        vectors_searched = demo_nprobe * vectors_per_cluster
        search_pct = (vectors_searched / demo_vectors) * 100
        # Recall roughly follows a curve based on nprobe
        estimated_recall = min(100, 50 + 50 * (1 - np.exp(-demo_nprobe / 3)))
        
        st.metric("Clusters searched", f"{demo_nprobe} of {demo_clusters}")
        st.metric("Vectors checked", f"{vectors_searched:,} of {demo_vectors:,}")
        st.metric("% of data searched", f"{search_pct:.1f}%")
    
    with nprobe_demo_cols[1]:
        # Visual representation
        st.markdown("**Cluster Search Visualization:**")
        
        # Create a visual grid of clusters
        cluster_html = '<div style="display: flex; flex-wrap: wrap; gap: 4px;">'
        for i in range(demo_clusters):
            if i < demo_nprobe:
                color = "#10b981"  # Green - searched
                label = "✓"
            else:
                color = "#374151"  # Gray - skipped
                label = ""
            cluster_html += f'<div style="width: 30px; height: 30px; background: {color}; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 12px;">{label}</div>'
        cluster_html += '</div>'
        
        st.markdown(cluster_html, unsafe_allow_html=True)
        st.caption(f"Green = searched ({demo_nprobe}), Gray = skipped ({demo_clusters - demo_nprobe})")
        
        # Show trade-off
        st.markdown(f"""
        **Estimated Performance:**
        - ⏱️ Speed: **{100 - search_pct:.0f}% faster** than Flat
        - 🎯 Recall: ~**{estimated_recall:.0f}%** of true results
        """)
    
    st.divider()
    
    # ===== SECTION 3: LIVE BENCHMARK =====
    st.markdown("### 🧪 Live Benchmark (Your Machine)")
    
    st.warning("""
    ⚠️ **Note:** With small datasets (< 10K), you won't see dramatic speed differences - 
    that's expected! The overhead of building an index can exceed the search savings. 
    This benchmark demonstrates the concepts; production systems use 100K+ vectors.
    """)
    
    # Benchmark settings
    bench_cols = st.columns(3)
    with bench_cols[0]:
        bench_vectors = st.selectbox("Database size", [1000, 5000, 10000, 20000], index=1)
    with bench_cols[1]:
        bench_dims = st.selectbox("Dimensions", [50, 100, 200], index=0)
    with bench_cols[2]:
        bench_queries = st.selectbox("Queries to run", [20, 50, 100], index=0)
    
    if st.button("🏃 Run Benchmark", type="primary"):
        # Create test data
        test_vectors, test_labels = create_sample_data(bench_vectors, bench_dims)
        test_ids = [f"doc_{i}" for i in range(bench_vectors)]
        
        # Calculate appropriate parameters
        num_clusters = max(10, int(np.sqrt(bench_vectors)))
        
        results_data = []
        
        progress = st.progress(0)
        status = st.empty()
        
        # First, build flat DB for ground truth
        status.text("Building ground truth (Flat)...")
        flat_db = SimpleVectorDB(strategy='flat', dimensions=bench_dims)
        flat_db.add_batch(test_ids, test_vectors)
        
        strategies_config = [
            ('flat', {}, "📖 Flat", "Exact search - checks all vectors"),
            ('ivf', {'num_clusters': num_clusters, 'nprobe': 1}, f"📊 IVF (nprobe=1)", f"Searches 1/{num_clusters} of data"),
            ('ivf', {'num_clusters': num_clusters, 'nprobe': 5}, f"📊 IVF (nprobe=5)", f"Searches 5/{num_clusters} of data"),
            ('ivf', {'num_clusters': num_clusters, 'nprobe': 10}, f"📊 IVF (nprobe=10)", f"Searches 10/{num_clusters} of data"),
        ]
        
        query_indices = np.random.choice(bench_vectors, bench_queries, replace=False)
        
        for i, (strat, kwargs, name, desc) in enumerate(strategies_config):
            status.text(f"Testing {name}...")
            progress.progress((i + 1) / len(strategies_config))
            
            # Create and populate DB
            db = SimpleVectorDB(strategy=strat, dimensions=bench_dims, **kwargs)
            db.add_batch(test_ids, test_vectors)
            db.build_index()
            
            # Run queries and measure ONLY search time (not ground truth computation)
            search_times = []
            total_recall = 0
            
            for idx in query_indices:
                query = test_vectors[idx]
                
                # Time just the search
                start = time.perf_counter()
                results = db.search(query, top_k=10)
                search_times.append(time.perf_counter() - start)
                
                # Compute recall against flat
                if strat != 'flat':
                    flat_results = flat_db.search(query, top_k=10)
                    flat_ids = set(id for id, _ in flat_results)
                    result_ids = set(id for id, _ in results)
                    recall = len(flat_ids & result_ids) / 10
                    total_recall += recall
                else:
                    total_recall += 1.0
            
            avg_search_ms = np.mean(search_times) * 1000
            avg_recall = total_recall / bench_queries
            
            results_data.append({
                'Strategy': name,
                'Description': desc,
                'Avg Search (ms)': avg_search_ms,
                'Recall @10': avg_recall * 100,
            })
        
        progress.empty()
        status.empty()
        
        # Show results
        st.markdown("### 📊 Benchmark Results")
        
        # Create comparison table
        df = pd.DataFrame(results_data)
        
        # Add speedup column
        flat_time = df[df['Strategy'] == '📖 Flat']['Avg Search (ms)'].values[0]
        df['vs Flat'] = df['Avg Search (ms)'].apply(lambda x: f"{flat_time/x:.1f}x" if x > 0 else "1.0x")
        
        # Format for display
        display_df = df.copy()
        display_df['Avg Search (ms)'] = display_df['Avg Search (ms)'].apply(lambda x: f"{x:.3f}")
        display_df['Recall @10'] = display_df['Recall @10'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df[['Strategy', 'Description', 'Avg Search (ms)', 'Recall @10', 'vs Flat']], 
                    use_container_width=True, hide_index=True)
        
        st.markdown("""
        **How to read this:**
        - **Avg Search**: Time per query in milliseconds (lower = faster)
        - **Recall @10**: What % of true top-10 results were found (higher = better)
        - **vs Flat**: Speedup compared to brute force (higher = faster)
        
        **Key insight:** Notice how IVF with nprobe=1 is fastest but may miss results, 
        while nprobe=10 is slower but finds almost everything!
        """)
    
    st.divider()
    
    # Summary table
    st.markdown("### 💡 When to Use Each Strategy")
    
    import pandas as pd
    
    summary = pd.DataFrame({
        "Aspect": [
            "Time Complexity",
            "Accuracy",
            "Memory",
            "Best For",
            "Real-world Example"
        ],
        "📖 Flat": [
            "O(n)",
            "100% (exact)",
            "Low",
            "< 10K vectors",
            "Small FAQ search"
        ],
        "🎲 LSH": [
            "O(1) ~ O(n)",
            "~80-95%",
            "Medium",
            "Very high dimensions",
            "Image similarity"
        ],
        "📊 IVF": [
            "O(k × cluster_size)",
            "~90-99%",
            "Medium",
            "100K - 10M vectors",
            "Document search"
        ]
    })
    
    st.dataframe(summary, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### 🔑 Key Takeaway
    
    > Vector databases solve the **"needle in a haystack"** problem for AI.
    > They trade a bit of accuracy for massive speedups, making
    > semantic search practical at scale!
    
    **Coming up in Workshop 4:** Attention - how does the model know what to focus on?
    """)


# ============================================================================
# TAB 4: Production Qdrant Demo
# ============================================================================
with tab4:
    st.markdown("## 🚀 Production Vector Database: Qdrant")
    
    st.markdown("""
    Our `SimpleVectorDB` teaches the concepts. **Qdrant** is what you'd use in production!
    
    | Feature | SimpleVectorDB | Qdrant |
    |---------|---------------|--------|
    | **Purpose** | Learning | Production |
    | **Language** | Pure Python | Rust (fast!) |
    | **Algorithm** | Flat, LSH, IVF | HNSW (graph-based) |
    | **Scale** | ~100K vectors | Billions |
    | **Features** | Basic search | Filtering, persistence, clustering, REST API |
    """)
    
    st.divider()
    
    if not QDRANT_AVAILABLE:
        st.warning("""
        ⚠️ **Qdrant not installed.** To try the live demo, run:
        ```
        pip install qdrant-client
        ```
        Then restart the app.
        """)
        
        st.markdown("### 📝 What Qdrant Code Looks Like")
        
        st.code('''
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Create in-memory client (or connect to server)
client = QdrantClient(":memory:")

# Create a collection (like a table)
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=100, distance=Distance.COSINE)
)

# Add vectors with metadata
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(id=1, vector=[0.1, 0.2, ...], payload={"title": "Doc 1", "category": "tech"}),
        PointStruct(id=2, vector=[0.15, 0.25, ...], payload={"title": "Doc 2", "category": "science"}),
    ]
)

# Search with optional filtering!
results = client.search(
    collection_name="documents",
    query_vector=[0.12, 0.22, ...],
    query_filter={"must": [{"key": "category", "match": {"value": "tech"}}]},
    limit=10
)
''', language='python')
        
        st.markdown("""
        ### 🔑 Key Qdrant Features
        
        1. **HNSW Algorithm**: Graph-based search, faster than IVF for most cases
        2. **Metadata Filtering**: Search within categories, date ranges, etc.
        3. **Persistence**: Data survives restarts
        4. **REST/gRPC API**: Use from any language
        5. **Cloud Option**: Managed hosting available
        """)
    
    else:
        st.success("✅ Qdrant is installed! Let's compare it to our SimpleVectorDB.")
        
        st.markdown("### 🏃 Live Benchmark: SimpleVectorDB vs Qdrant")
        
        qdrant_cols = st.columns(3)
        with qdrant_cols[0]:
            qdrant_vectors = st.selectbox("Vectors", [1000, 5000, 10000], index=1, key="qdrant_vectors")
        with qdrant_cols[1]:
            qdrant_dims = st.selectbox("Dimensions", [50, 100, 200], index=1, key="qdrant_dims")
        with qdrant_cols[2]:
            qdrant_queries = st.selectbox("Queries", [50, 100, 200], index=0, key="qdrant_queries")
        
        if st.button("🏃 Run Qdrant Comparison", type="primary"):
            # Generate test data
            np.random.seed(42)
            test_vectors = np.random.randn(qdrant_vectors, qdrant_dims).astype(np.float32)
            test_vectors = test_vectors / (np.linalg.norm(test_vectors, axis=1, keepdims=True) + 1e-10)
            query_indices = np.random.choice(qdrant_vectors, qdrant_queries, replace=False)
            
            results_data = []
            progress = st.progress(0)
            status = st.empty()
            
            # Test 1: SimpleVectorDB Flat
            status.text("Testing SimpleVectorDB (Flat)...")
            progress.progress(0.2)
            
            db_flat = SimpleVectorDB(strategy='flat', dimensions=qdrant_dims)
            ids = [f"doc_{i}" for i in range(qdrant_vectors)]
            db_flat.add_batch(ids, test_vectors)
            
            start = time.perf_counter()
            for idx in query_indices:
                db_flat.search(test_vectors[idx], top_k=10)
            flat_time = (time.perf_counter() - start) * 1000 / qdrant_queries
            
            results_data.append({
                'System': '📖 SimpleVectorDB (Flat)',
                'Avg Search (ms)': flat_time,
                'Algorithm': 'Brute Force',
            })
            
            # Test 2: SimpleVectorDB IVF
            status.text("Testing SimpleVectorDB (IVF)...")
            progress.progress(0.4)
            
            num_clusters = max(10, int(np.sqrt(qdrant_vectors)))
            db_ivf = SimpleVectorDB(strategy='ivf', dimensions=qdrant_dims, num_clusters=num_clusters, nprobe=5)
            db_ivf.add_batch(ids, test_vectors)
            db_ivf.build_index()
            
            start = time.perf_counter()
            for idx in query_indices:
                db_ivf.search(test_vectors[idx], top_k=10)
            ivf_time = (time.perf_counter() - start) * 1000 / qdrant_queries
            
            results_data.append({
                'System': '📊 SimpleVectorDB (IVF)',
                'Avg Search (ms)': ivf_time,
                'Algorithm': f'IVF ({num_clusters} clusters)',
            })
            
            # Test 3: Qdrant
            status.text("Testing Qdrant...")
            progress.progress(0.7)
            
            client = QdrantClient(":memory:")
            client.create_collection(
                collection_name="benchmark",
                vectors_config=VectorParams(size=qdrant_dims, distance=Distance.COSINE)
            )
            
            # Add vectors in batches
            batch_size = 500
            for i in range(0, qdrant_vectors, batch_size):
                end_idx = min(i + batch_size, qdrant_vectors)
                points = [
                    PointStruct(id=j, vector=test_vectors[j].tolist())
                    for j in range(i, end_idx)
                ]
                client.upsert(collection_name="benchmark", points=points)
            
            # Search
            start = time.perf_counter()
            for idx in query_indices:
                client.query_points(
                    collection_name="benchmark",
                    query=test_vectors[idx].tolist(),
                    limit=10
                )
            qdrant_time = (time.perf_counter() - start) * 1000 / qdrant_queries
            
            results_data.append({
                'System': '🚀 Qdrant',
                'Avg Search (ms)': qdrant_time,
                'Algorithm': 'HNSW (graph)',
            })
            
            progress.progress(1.0)
            status.empty()
            progress.empty()
            
            # Show results
            st.markdown("### 📊 Results")
            
            import pandas as pd
            df = pd.DataFrame(results_data)
            
            # Add speedup column
            flat_baseline = df[df['System'].str.contains('Flat')]['Avg Search (ms)'].values[0]
            df['vs Flat'] = df['Avg Search (ms)'].apply(lambda x: f"{flat_baseline/x:.1f}x faster" if x < flat_baseline else "baseline")
            
            # Format
            df['Avg Search (ms)'] = df['Avg Search (ms)'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Interpretation:**
            - Qdrant uses HNSW, a graph-based algorithm that's typically faster than IVF
            - At small scale, the differences may be small
            - At 100K+ vectors, Qdrant really shines!
            
            **Note:** Qdrant also supports:
            - Filtering (e.g., "only search documents from 2024")
            - Payload storage (metadata with each vector)
            - Distributed deployment for massive scale
            """)
        
        st.divider()
        
        st.markdown("### 💡 When to Use What")
        
        st.markdown("""
        | Scenario | Recommendation |
        |----------|---------------|
        | **Learning/Prototyping** | SimpleVectorDB - understand the concepts |
        | **Small project (< 10K vectors)** | SimpleVectorDB or Chroma |
        | **Production app** | Qdrant, Pinecone, or Weaviate |
        | **Massive scale (100M+ vectors)** | Qdrant Cloud, Pinecone, or Milvus |
        
        **Our workshop teaches you WHY these systems work, so you can use them effectively!**
        """)


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>📚 Workshop 3: Vector Databases | GenAI Self-Build Series</p>
    <p><em>The alien's library with magic shelves - finding similar things fast!</em></p>
</div>
""", unsafe_allow_html=True)
