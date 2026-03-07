// Workshop 3: Vector Databases
const VectorDBPage = {
    // We cache documents so they stay stable across re-renders
    documents: [
        "Python is a programming language",
        "Java is also a programming language",
        "Machine learning uses algorithms",
        "Deep learning is a subset of ML",
        "Cats are furry pets",
        "Dogs are loyal companions",
        "The weather is sunny today",
        "It might rain tomorrow"
    ],
    dbText: null,

    initDB() {
        if (!this.dbText && window.TextVectorDB) {
            this.dbText = new TextVectorDB(64);

            for (let i = 0; i < this.documents.length; i++) {
                const doc = this.documents[i];
                this.dbText.add(`doc_${i}`, TextUtils.makeVector(doc, 64));
            }
        }
    },

    render() {
        const isVisual = AppState.learningPath === "Visual";
        
        let content = `
            <div class="page-container">
                <div class="main-header">
                    <h2>📚 Workshop 3: Vector Databases</h2>
                    <h3><em>📘 Chapter 3 of Zara's Journey: The Library</em></h3>
                </div>
        `;

        if (!isVisual) {
            this.initDB();
            
            content += `
                <div class="alert alert-info">
                    <p><strong>> 🛸 Zara has millions of meaning vectors now. She needs a magic library where she can find the right one in seconds, not hours.</strong></p>
                    <p><strong>The Problem:</strong> With millions of text embeddings, how do we find similar ones fast?</p>
                    <p><strong>The Solution:</strong> Vector databases with smart indexing and similarity search.</p>
                </div>
                
                <p><strong>Database contains ${this.documents.length} documents</strong></p>
                
                <div class="card-grid">
                    <div>
                        <label for="vdb-query">Search query:</label>
                        <input type="text" id="vdb-query" class="text-input" value="programming languages">
                    </div>
                    <div>
                        <label for="vdb-k">Number of results:</label>
                        <input type="range" id="vdb-k" min="1" max="5" value="3" oninput="document.getElementById('vdb-k-val').innerText=this.value">
                        <span id="vdb-k-val">3</span>
                    </div>
                </div>
                
                <button id="vdb-search-btn" class="mega-button" style="padding: 15px; margin: 20px 0;">🔍 Search Text</button>
                
                <div id="vdb-results-container"></div>
            `;
        } else {
            // Visual Domain
            const scenes = Object.keys(SAMPLE_SCENES).map(k => `<option value="${k}">${k}</option>`).join("");
            
            content += `
                <div class="alert alert-info">
                    <p><strong>The Problem:</strong> Zara wants to find layout scenes that look like ones she's seen before, but checking them line-by-line is way too slow.</p>
                    <p><strong>The Solution:</strong> She stores the 16D vector representation of every scene in a Vector Database. When she sees a new scene, she queries the database for nearest neighbors purely by their coordinate distance!</p>
                    <p style="margin-top:8px; color: var(--accent-3);"><strong>🔌 Layout analogy:</strong> You're already a human vector database — flipping through old tape-outs looking for something "kind of like" your current design. This does it in milliseconds. Try searching from <em>inverter_cell</em> to find similar circuit layouts!</p>
                </div>
                
                <div class="card-grid" style="grid-template-columns: 1fr 2fr;">
                    <div>
                        <label for="vdb-scene-sel">Select a query scene:</label>
                        <select id="vdb-scene-sel" class="select-input" style="margin-bottom: 20px;">
                            ${scenes}
                        </select>
                        <p><strong>Query Target:</strong></p>
                        <div id="vdb-query-scene"></div>
                    </div>
                    
                    <div>
                        <h3>Top Visual Matches</h3>
                        <div id="vdb-visual-results" class="card-grid" style="grid-template-columns: repeat(3, 1fr); gap: 10px;"></div>
                    </div>
                </div>
            `;
        }
        
        content += `
            ${chapterNav("vectordb")}
        </div>`;
        return content;
    },

    attachEvents() {
        const isVisual = AppState.learningPath === "Visual";
        
        if (!isVisual) {
            const btn = document.getElementById("vdb-search-btn");
            btn.addEventListener("click", () => this.runTextSearch());
            // Run empty default search on load for UX
            this.runTextSearch();
        } else {
            const sceneSel = document.getElementById("vdb-scene-sel");
            sceneSel.addEventListener("change", () => this.runVisualSearch(sceneSel.value));
            this.runVisualSearch(sceneSel.value);
        }
    },

    runTextSearch() {
        if (!this.dbText) return;
        
        const query = document.getElementById("vdb-query").value;
        const k = parseInt(document.getElementById("vdb-k").value, 10);
        const normalizedQVec = TextUtils.makeVector(query, 64);
        
        // Search
        const results = this.dbText.search(normalizedQVec, k);
        
        let html = `<h3>Results:</h3>`;
        results.forEach((res, i) => {
            const docIdx = parseInt(res.id.split('_')[1], 10);
            html += `
                <div class="workshop-card" style="margin-bottom: 15px;">
                    <strong>#${i+1}</strong> (Score: ${res.score.toFixed(4)})<br>
                    ${this.documents[docIdx]}
                </div>
            `;
        });
        
        document.getElementById("vdb-results-container").innerHTML = html;
    },

    runVisualSearch(sceneName) {
        const scene = SAMPLE_SCENES[sceneName];
        if (!scene || !AppState.engines.db || !AppState.engines.embedding) return;

        // Render input
        document.getElementById("vdb-query-scene").innerHTML = TextUtils.renderSceneHtml(scene, 30, 200);

        // Get embedding
        const qEmb = AppState.engines.embedding.get_embedding(scene);
        
        // Search
        const results = AppState.engines.db.search(qEmb, 4);
        
        // Filter out exact match 
        const filtered = results.filter(r => r[0] !== sceneName).slice(0, 3);
        
        let html = "";
        filtered.forEach(res => {
            const rName = res[0];
            const rScore = res[1];
            const rScene = SAMPLE_SCENES[rName];
            
            html += `
                <div>
                    <strong>${rName}</strong><br>
                    <span style='color:var(--accent-1);font-size:12px;'>Sim: ${rScore.toFixed(2)}</span>
                    ${TextUtils.renderSceneHtml(rScene, 15, 150)}
                </div>
            `;
        });
        
        document.getElementById("vdb-visual-results").innerHTML = html;
    }
};
