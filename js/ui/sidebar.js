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
                <a href="#home" class="nav-link" data-route="home">🏠 Home: Zara's Arrival</a>
                <a href="#tokenization" class="nav-link" data-route="tokenization">📕 Ch 1: The Codebook</a>
                <a href="#embeddings" class="nav-link" data-route="embeddings">📗 Ch 2: The Map</a>
                <a href="#vectordb" class="nav-link" data-route="vectordb">📘 Ch 3: The Library</a>
                <a href="#attention" class="nav-link" data-route="attention">📙 Ch 4: The Spotlight</a>
                <a href="#transformers" class="nav-link" data-route="transformers">📓 Ch 5: The Brain</a>
                <a href="#rag" class="nav-link" data-route="rag">📔 Ch 6: The Search Engine</a>
                <a href="#pipeline" class="nav-link" data-route="pipeline">🔗 Finale: The Complete System</a>
            </nav>
            
            <hr>
            
            <div class="sidebar-section">
                <h3>🛤️ Select Path</h3>
                <div class="path-toggle">
                    <label>
                        <input type="radio" name="path" value="Text" ${!isVisual ? 'checked' : ''} 
                               onchange="AppState.setLearningPath('Text')">
                        📝 Learn with Words
                    </label>
                    <label>
                        <input type="radio" name="path" value="Visual" ${isVisual ? 'checked' : ''}
                               onchange="AppState.setLearningPath('Visual')">
                        🎨 Learn with Shapes
                    </label>
                </div>
            </div>
            
            <hr>
            
            <div class="sidebar-section">
                <h3>🛸 Meet Zara</h3>
                <p>Zara is a Zorathian scientist who crash-lands on Earth. She can't understand human language OR visual scenes. Follow her journey as she builds every component of an AI system — tackling words in one hand and shapes in the other.</p>
                <p style="font-size: 0.85rem; color: var(--text-secondary);">Two paths. One protagonist. One complete AI journey.</p>
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
