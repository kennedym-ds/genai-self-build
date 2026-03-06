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
