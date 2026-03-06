const Sidebar = {
    container: null,

    init() {
        this.container = document.getElementById("sidebar-container");
    },

    render() {
        if (!this.container) this.init();

        const isVisual = AppState.learningPath === "Visual";
        
        this.container.innerHTML = `
            <div class="sidebar-header">
                <h2>🗺️ Workshop Navigator</h2>
            </div>
            <nav class="sidebar-nav">
                <a href="#home" class="nav-link" data-route="home">🏠 Home: The Big Picture</a>
                <a href="#tokenization" class="nav-link" data-route="tokenization">1️⃣ Tokenization</a>
                <a href="#embeddings" class="nav-link" data-route="embeddings">2️⃣ Embeddings</a>
                <a href="#vectordb" class="nav-link" data-route="vectordb">3️⃣ Vector Databases</a>
                <a href="#attention" class="nav-link" data-route="attention">4️⃣ Attention</a>
                <a href="#transformers" class="nav-link" data-route="transformers">5️⃣ Transformers</a>
                <a href="#rag" class="nav-link" data-route="rag">6️⃣ RAG</a>
                <a href="#pipeline" class="nav-link" data-route="pipeline">🔗 End-to-End Pipeline</a>
            </nav>
            
            <hr>
            
            <div class="sidebar-section">
                <h3>🛤️ Select Path</h3>
                <div class="path-toggle">
                    <label>
                        <input type="radio" name="path" value="Text" ${!isVisual ? 'checked' : ''} 
                               onchange="AppState.setLearningPath('Text')">
                        📝 Text Domain (Words)
                    </label>
                    <label>
                        <input type="radio" name="path" value="Visual" ${isVisual ? 'checked' : ''}
                               onchange="AppState.setLearningPath('Visual')">
                        🛸 Visual Domain (Zara)
                    </label>
                </div>
            </div>
            
            <hr>
            
            <div class="sidebar-section">
                <h3>🛸 Meet Zara</h3>
                <p>Zara is a Zorathian scientist orbiting Earth. She can intercept human text — but can't understand a word.</p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">Follow her journey through 6 chapters as she builds every component of a modern AI system from scratch.</p>
            </div>
        `;
        
        this.setActiveLink(AppState.activeRoute);
    },

    setActiveLink(route) {
        if (!this.container) return;
        
        const links = this.container.querySelectorAll('.nav-link');
        links.forEach(link => {
            if (link.dataset.route === route) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
};
