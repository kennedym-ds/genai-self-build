function escapeHtml(value) {
    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

window.escapeHtml = escapeHtml;

const CHAPTERS = [
    { route: "home", label: "🏠 Home", title: "Zara's Arrival" },
    { route: "tokenization", label: "📕 Ch 1", title: "The Codebook" },
    { route: "embeddings", label: "📗 Ch 2", title: "The Map" },
    { route: "vectordb", label: "📘 Ch 3", title: "The Library" },
    { route: "attention", label: "📙 Ch 4", title: "The Spotlight" },
    { route: "transformers", label: "📓 Ch 5", title: "The Brain" },
    { route: "rag", label: "📔 Ch 6", title: "The Search Engine" },
    { route: "pipeline", label: "🔗 Finale", title: "The Complete System" }
];

const CHAPTER_TRANSITIONS = {
    tokenization: "Zara can read now — she's cracked the codebook! But Token 42 and Token 43 look identical to her. She needs to discover that words have MEANING, and similar meanings should live nearby...",
    embeddings: "Zara's map of meaning is growing — words cluster beautifully! But she has millions of vectors now. Searching through them one by one? That would take forever. She needs a smarter library...",
    vectordb: "Zara can find anything in her library in milliseconds! But she keeps making the same mistake — 'bank' means the same thing whether it's near 'river' or 'money.' She needs to learn that CONTEXT changes everything...",
    attention: "The spotlight changed everything. Zara can now see how words shape each other's meaning. She has all the pieces: a codebook, a map, a library, and spotlights. Time to assemble them into a complete brain...",
    transformers: "Zara's brain WORKS — it can process any input! But there's a problem: it sometimes makes things up. It hallucinates. She needs a way to CHECK her answers against real knowledge...",
    rag: "Zara can now look things up before answering — no more hallucinations! She's built every piece. Time to see them all work together as ONE system..."
};

function chapterNav(currentRoute) {
    const currentIdx = CHAPTERS.findIndex(chapter => chapter.route === currentRoute);
    if (currentIdx === -1) {
        return "";
    }

    const previous = currentIdx > 0 ? CHAPTERS[currentIdx - 1] : null;
    const next = currentIdx < CHAPTERS.length - 1 ? CHAPTERS[currentIdx + 1] : null;

    const showPrev = Boolean(previous) && currentRoute !== "tokenization";
    const showNext = Boolean(next) && currentRoute !== "pipeline";
    const transition = CHAPTER_TRANSITIONS[currentRoute] || "";

    const transitionHtml = transition
        ? `<p style="margin: 0 0 18px 0; color: var(--text-secondary); font-size: 1rem;"><em>\"${escapeHtml(transition)}\"</em></p>`
        : "";

    const prevHtml = showPrev
        ? `<a href="#${previous.route}" class="mega-button" style="display:block; width:auto; flex:1; min-width:260px; padding:14px 18px; text-decoration:none; text-align:left;">← Previous Chapter<br><small>${escapeHtml(previous.label)}: ${escapeHtml(previous.title)}</small></a>`
        : "";

    const nextHtml = showNext
        ? `<a href="#${next.route}" class="mega-button" style="display:block; width:auto; flex:1; min-width:260px; padding:14px 18px; text-decoration:none; text-align:right;">Next Chapter →<br><small>${escapeHtml(next.label)}: ${escapeHtml(next.title)}</small></a>`
        : "";

    return `
        <hr>
        ${transitionHtml}
        <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:stretch;">
            ${prevHtml}
            ${nextHtml}
        </div>
    `;
}

window.CHAPTERS = CHAPTERS;
window.chapterNav = chapterNav;

// Global App State
const AppState = {
    learningPath: "Text", // "Text" | "Visual"
    activeRoute: "home",
    
    // ML Engines (initialized on DOM load)
    engines: {
        tokenizer: null,
        embedding: null,
        db: null,
        attention: null
    },

    setLearningPath(path) {
        this.learningPath = path;
        // Re-render current page to reflect new domain
        Router.navigate(this.activeRoute, true);
        // Update sidebar UI
        Sidebar.render();
    }
};

// Route name → Page object mapping
const PAGE_MAP = {
    "home": HomePage,
    "tokenization": TokenizationPage,
    "embeddings": EmbeddingsPage,
    "vectordb": VectorDBPage,
    "attention": AttentionPage,
    "transformers": TransformersPage,
    "rag": RAGPage,
    "pipeline": PipelinePage
};

// Simple Hash Router
const Router = {
    init() {
        window.addEventListener("hashchange", () => this.handleHash());
        this.handleHash();
    },

    handleHash() {
        let hash = window.location.hash.replace("#", "") || "home";
        this.navigate(hash);
    },

    navigate(route, force = false) {
        if (!PAGE_MAP[route]) {
            console.warn("Unknown route:", route, "— falling back to home");
            route = "home";
        }
        
        if (AppState.activeRoute !== route || force) {
            AppState.activeRoute = route;
            const contentContainer = document.getElementById("main-content");
            
            // Render the page HTML
            contentContainer.innerHTML = PAGE_MAP[route].render();
            
            // Attach event listeners if the page defines them
            if (PAGE_MAP[route].attachEvents) {
                PAGE_MAP[route].attachEvents();
            }
            
            Sidebar.setActiveLink(route);
            
            // Scroll to top on navigation
            window.scrollTo(0, 0);
        }
    }
};

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Shape Engines for Visual Path
    if (typeof ShapeTokenizer !== 'undefined') {
        AppState.engines.tokenizer = new ShapeTokenizer('part');
        AppState.engines.tokenizer.train(Object.values(SAMPLE_SCENES));
        
        AppState.engines.embedding = new ShapeEmbedding('spatial', 16);
        AppState.engines.embedding.train(SAMPLE_SCENES);
        
        AppState.engines.db = new ShapeVectorDB(16);
        AppState.engines.db.add_batch(
            AppState.engines.embedding.scene_names, 
            AppState.engines.embedding.embeddings
        );
        
        AppState.engines.attention = new ShapeAttention();
    }

    // 2. Render Shell
    Sidebar.render();
    
    // 3. Start Router
    Router.init();
});
