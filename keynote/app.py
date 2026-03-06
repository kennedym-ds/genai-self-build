"""
🛸 Zara Crash-Lands in Shape World — Keynote Demo App
=====================================================

Four interactive demo tabs that power the live keynote moments:
  Tab 1: Shape Tokenizer   (Act 1 — "What Am I Looking At?")
  Tab 2: Meaning Map        (Act 2 — "Do These Go Together?")
  Tab 3: Shape Library      (Act 2 — search by similarity)
  Tab 4: Attention Heatmap  (Act 3 — "Context Matters!")

Run with:
    cd keynote
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import sys
import os

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from shape_data import (
    SHAPE_TYPES, LAYER_COLORS, SHAPE_PARTS, SHAPE_DECOMPOSITION,
    SAMPLE_SCENES, scene_to_feature_vector,
)
from shapes_core import ShapeTokenizer, ShapeEmbedding, ShapeVectorDB, ShapeAttention

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="🛸 Zara's Shape World",
    page_icon="🛸",
    layout="wide",
)

# =========================================================================
# CUSTOM CSS
# =========================================================================
st.markdown("""
<style>
    .token-box {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 13px;
        color: white;
    }
    .tok-pixel  { background-color: #3b82f6; }
    .tok-shape  { background-color: #8b5cf6; }
    .tok-part   { background-color: #10b981; }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #60a5fa; }
    .metric-label { font-size: 12px; color: #94a3b8; }
    .alien-box {
        background-color: #1e1e2e;
        border: 2px solid #7c3aed;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .heatmap-cell {
        display: inline-block;
        width: 48px;
        height: 48px;
        margin: 1px;
        text-align: center;
        line-height: 48px;
        font-size: 10px;
        border-radius: 4px;
    }
    .scene-shape {
        position: absolute;
        border: 2px solid rgba(255,255,255,0.3);
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    .scene-canvas {
        position: relative;
        background: #0f172a;
        border: 2px solid #334155;
        border-radius: 8px;
        width: 400px;
        height: 400px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# HEADER
# =========================================================================
st.title("🛸 Zara Crash-Lands in Shape World")
st.markdown("### *An alien learns to see — and accidentally explains GenAI*")

# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("## 🛸 Meet Zara")
    st.markdown("""
    Zara comes from a world of **pure sound**.
    She has never seen a shape.

    She crash-lands on Earth, opens her eyes, and sees...
    **chaos**. Colors, edges, curves — all at once.

    Follow her journey through 4 demo tabs to see
    how she learns to understand what she sees.
    """)
    st.divider()
    st.markdown("**Keynote Demo App**")
    st.markdown("Michael Kennedy · Analog Layout Conference")
    st.divider()
    scene_names = list(SAMPLE_SCENES.keys())
    selected_scene = st.selectbox("🎨 Choose a scene", scene_names, index=0)


# =========================================================================
# HELPERS
# =========================================================================

def render_scene_html(scene, scale: float = 40, canvas_px: int = 400):
    """Render a scene as an HTML overlay of colored boxes."""
    html = f'<div style="position:relative;background:#0f172a;border:2px solid #334155;border-radius:8px;width:{canvas_px}px;height:{canvas_px}px;overflow:hidden;">'
    for s in scene["shapes"]:
        color_hex = LAYER_COLORS.get(s["color"], "#888888")
        left = s["x"] * scale
        top = s["y"] * scale
        w = s["w"] * scale
        h = s["h"] * scale
        html += (
            f'<div style="position:absolute;left:{left}px;top:{top}px;'
            f'width:{w}px;height:{h}px;background:{color_hex}80;'
            f'border:2px solid {color_hex};border-radius:4px;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:10px;color:white;text-shadow:1px 1px 2px #000;">'
            f'{s["label"]}</div>'
        )
    html += '</div>'
    return html


def render_tokens_html(tokens: list, css_class: str) -> str:
    """Render a list of token strings as colored boxes."""
    boxes = "".join(f'<span class="token-box {css_class}">{t}</span>' for t in tokens)
    return f'<div style="line-height:2.2">{boxes}</div>'


# =========================================================================
# DATA INIT — cached so it only runs once
# =========================================================================

@st.cache_resource
def init_engines():
    """Train all engines once."""
    scenes = SAMPLE_SCENES

    # Tokenizers
    tok_pixel = ShapeTokenizer(strategy='pixel')
    tok_shape = ShapeTokenizer(strategy='shape')
    tok_part = ShapeTokenizer(strategy='part')
    all_scenes = list(scenes.values())
    tok_pixel.train(all_scenes)
    tok_shape.train(all_scenes)
    tok_part.train(all_scenes)

    # Embeddings
    emb_random = ShapeEmbedding(strategy='random', dimensions=16)
    emb_spatial = ShapeEmbedding(strategy='spatial', dimensions=16)
    emb_feature = ShapeEmbedding(strategy='feature', dimensions=16)
    emb_random.train(scenes)
    emb_spatial.train(scenes)
    emb_feature.train(scenes)

    # Vector DB (using spatial embeddings)
    db = ShapeVectorDB(dimensions=16)
    db.add_batch(emb_spatial.scene_names, emb_spatial.embeddings)

    # Attention
    att = ShapeAttention()

    return {
        "tokenizers": {"pixel": tok_pixel, "shape": tok_shape, "part": tok_part},
        "embeddings": {"random": emb_random, "spatial": emb_spatial, "feature": emb_feature},
        "db": db,
        "attention": att,
        "emb_spatial": emb_spatial,
    }


engines = init_engines()
scene = SAMPLE_SCENES[selected_scene]


# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧩 Shape Tokenizer",
    "🗺️ Meaning Map",
    "📚 Shape Library",
    "👀 Attention Heatmap",
])


# =========================================================================
# TAB 1 — SHAPE TOKENIZER
# =========================================================================
with tab1:
    st.header("🧩 Act 1: What Am I Looking At?")
    st.markdown("""
    > *Zara is overwhelmed. She needs to break what she sees into manageable pieces.*

    Three strategies — from raw pixels to smart shape-parts:
    """)

    col_scene, col_tokens = st.columns([1, 2])

    with col_scene:
        st.markdown(f"**Scene: {selected_scene}**")
        st.markdown(render_scene_html(scene), unsafe_allow_html=True)

    with col_tokens:
        for strategy, css in [("pixel", "tok-pixel"), ("shape", "tok-shape"), ("part", "tok-part")]:
            tok = engines["tokenizers"][strategy]
            ids = tok.encode(scene)
            labels = tok.decode(ids)

            emoji = {"pixel": "🔲", "shape": "📦", "part": "🧩"}[strategy]
            analogy = {
                "pixel": "Every pixel — 'one air molecule at a time'",
                "shape": "Whole shapes — compact but brittle",
                "part": "Sub-parts — flexible alphabet (like BPE!)",
            }[strategy]

            st.markdown(f"**{emoji} {strategy.title()}** — {analogy}")

            # Show token boxes
            display_labels = labels[:60]  # cap for readability
            st.markdown(render_tokens_html(display_labels, css), unsafe_allow_html=True)

            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Tokens", len(ids))
            mcol2.metric("Vocab size", tok.vocab_size)
            st.markdown("---")

    # Connection callout
    st.info(
        "💡 **Connection to real AI**: When you upload a photo to ChatGPT, it gets chopped into "
        "16×16 pixel patches — each becomes a token. That's ViT (Vision Transformer) tokenization. "
        "The 'part' strategy here is analogous to VQ-VAE's learned visual codebook."
    )


# =========================================================================
# TAB 2 — MEANING MAP
# =========================================================================
with tab2:
    st.header("🗺️ Act 2: Do These Go Together?")
    st.markdown("""
    > *Zara gave each part a number. But the numbers are meaningless. She needs a map where
    > similar shapes live close together — and meaning becomes distance.*
    """)

    emb_strategy = st.radio(
        "Embedding strategy",
        ["random", "spatial", "feature"],
        index=1,
        horizontal=True,
        help="Random = no meaning. Spatial = co-occurrence. Feature = engineered properties.",
    )
    emb = engines["embeddings"][emb_strategy]

    col_map, col_sim = st.columns([3, 2])

    with col_map:
        st.markdown("**2D Meaning Map** (first 2 principal components)")
        proj = emb.get_2d_projection()

        # Build a simple scatter plot with matplotlib
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        xs, ys = proj[:, 0], proj[:, 1]
        highlight_idx = emb.scene_names.index(selected_scene)

        # All points
        ax.scatter(xs, ys, c="#60a5fa", s=60, alpha=0.6, edgecolors="white", linewidth=0.5)

        # Highlight selected
        ax.scatter(
            xs[highlight_idx], ys[highlight_idx],
            c="#f59e0b", s=200, marker="*", edgecolors="white", linewidth=1, zorder=5
        )

        # Labels
        for i, name in enumerate(emb.scene_names):
            ax.annotate(
                name, (xs[i], ys[i]),
                fontsize=7, color="white", alpha=0.85,
                textcoords="offset points", xytext=(5, 5),
            )

        ax.set_title(f"Shape Embeddings ({emb_strategy})", color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        st.pyplot(fig)
        plt.close(fig)

    with col_sim:
        st.markdown(f"**Most similar to '{selected_scene}'**")
        similar = emb.most_similar(selected_scene, top_k=5)
        for rank, (name, score) in enumerate(similar, 1):
            bar_width = max(0, int(score * 100))
            st.markdown(
                f"**{rank}.** {name} — `{score:.3f}`"
            )
            st.progress(max(0.0, min(1.0, (score + 1) / 2)))

        st.divider()
        st.markdown("**Embedding vector** (first 8 dims)")
        vec = emb.get_embedding(selected_scene)
        cols = st.columns(4)
        for i, v in enumerate(vec[:8]):
            cols[i % 4].metric(f"d{i}", f"{v:.2f}")

    st.info(
        "💡 **Connection to real AI**: This is exactly how CLIP works — images and text are "
        "projected into the same vector space. 'Meaning' becomes distance. Similar things cluster."
    )


# =========================================================================
# TAB 3 — SHAPE LIBRARY (VECTOR DB)
# =========================================================================
with tab3:
    st.header("📚 Act 2 (cont.): The Shape Library")
    st.markdown("""
    > *Zara has thousands of patterns. When she sees something new, she asks:
    > "Have I seen something like this before?" She needs a library where you
    > search by similarity, not by name.*
    """)

    db = engines["db"]
    emb_spatial = engines["emb_spatial"]

    st.markdown(f"**Library size:** {db.size} patterns")
    st.markdown(f"**Query:** {selected_scene}")

    query_vec = emb_spatial.get_embedding(selected_scene)
    results = db.search(query_vec, top_k=6)

    st.markdown("### Search Results (by similarity)")

    cols = st.columns(3)
    for i, (name, score) in enumerate(results):
        with cols[i % 3]:
            is_query = (name == selected_scene)
            border_color = "#f59e0b" if is_query else "#334155"
            st.markdown(
                f'<div style="border:2px solid {border_color};border-radius:8px;'
                f'padding:8px;margin:4px 0;text-align:center;">'
                f'<b>{"⭐ " if is_query else ""}{name}</b><br>'
                f'<span style="color:#60a5fa;">similarity: {score:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            result_scene = SAMPLE_SCENES.get(name)
            if result_scene:
                st.markdown(render_scene_html(result_scene, scale=30, canvas_px=260), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("*Raise your hand if you've ever flipped through old tape-outs looking for "
                "something 'kind of like' what you're working on now.*")
    st.info("💡 **You're a human vector database.** "
            "Google Photos, Spotify recommendations, and semantic code search all use the same principle.")


# =========================================================================
# TAB 4 — ATTENTION HEATMAP
# =========================================================================
with tab4:
    st.header("👀 Act 3: Wait, Context Matters!")
    st.markdown("""
    > *Zara sees a red circle. But she treats it the same whether it's on a traffic light or a
    > clown's nose. She needs to learn what to focus on — which shapes relate to each other.*
    """)

    att = engines["attention"]
    weights, labels = att.compute_attention(scene)

    if len(labels) == 0:
        st.warning("This scene has no shapes to compute attention for.")
    else:
        # Head selector
        head_options = ["All (combined)"] + [f"Head {i}: {ShapeAttention.HEAD_NAMES[i]}" for i in range(4)]
        head_choice = st.radio("Spotlight head", head_options, horizontal=True)

        if head_choice == "All (combined)":
            combined, _ = att.combined_attention(scene)
            display_weights = combined
            head_desc = "Average across all four spotlights"
        else:
            head_idx = head_options.index(head_choice) - 1
            display_weights = weights[head_idx]
            head_desc = {
                0: "🔦 **Proximity** — shapes that are physically close attend to each other",
                1: "🎨 **Color Match** — same-color shapes light up together",
                2: "📐 **Alignment** — shapes sharing x or y center lines",
                3: "📦 **Containment** — shapes nested inside others",
            }[head_idx]

        st.markdown(head_desc)

        col_scene2, col_heat = st.columns([1, 2])

        with col_scene2:
            st.markdown(f"**Scene: {selected_scene}**")
            st.markdown(render_scene_html(scene, scale=35, canvas_px=350), unsafe_allow_html=True)

        with col_heat:
            st.markdown("**Attention Heatmap**")
            n = len(labels)

            # Build heatmap with matplotlib
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")
            fig, ax = plt.subplots(figsize=(max(4, n * 0.7), max(3, n * 0.6)))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")

            im = ax.imshow(display_weights, cmap="YlOrRd", aspect="equal", vmin=0)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="white")
            ax.set_yticklabels(labels, fontsize=8, color="white")
            ax.set_xlabel("Attends TO →", color="white", fontsize=10)
            ax.set_ylabel("← Shape", color="white", fontsize=10)

            # Annotate cells
            for i in range(n):
                for j in range(n):
                    val = display_weights[i, j]
                    text_color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Focus shape picker
        st.markdown("---")
        focus_shape = st.selectbox("🔍 Focus on a shape", labels)
        if focus_shape:
            idx = labels.index(focus_shape)
            row = display_weights[idx]
            st.markdown(f"**'{focus_shape}' pays attention to:**")
            ranked = np.argsort(-row)
            for r in ranked:
                if labels[r] != focus_shape:
                    bar_val = float(row[r])
                    st.markdown(f"→ **{labels[r]}**: {bar_val:.3f}")
                    st.progress(min(1.0, bar_val))

    st.info(
        "💡 **Connection to real AI**: This is the self-attention mechanism from 'Attention Is All You Need' (2017). "
        "Multi-head attention lets the model learn different relationship types simultaneously — "
        "just like Zara's four spotlights."
    )


# =========================================================================
# FOOTER
# =========================================================================
st.markdown("---")
st.markdown(
    "*🛸 Zara Crash-Lands in Shape World — Keynote Demo · "
    "Part of the [GenAI Self-Build Workshop Series](../README.md)*"
)
