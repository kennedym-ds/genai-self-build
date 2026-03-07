// Workshop 2: Embeddings
const EmbeddingsPage = {
    render() {
        const isVisual = AppState.learningPath === "Visual";
        
        let content = `
            <div class="page-container">
                <div class="main-header">
                    <h2>🗺️ Workshop 2: Embeddings</h2>
                    <h3><em>📗 Chapter 2 of Zara's Journey: The Map</em></h3>
                </div>
        `;

        if (!isVisual) {
            content += `
                <div class="alert alert-info">
                    <p><strong>> 🛸 Zara can read symbols now, but Token 42 ('cat') and Token 43 ('dog') look the same to her. She needs a map where similar meanings live nearby.</strong></p>
                    <p><strong>The Problem:</strong> Token IDs don't capture meaning. "Cat" (ID 42) and "Dog" (ID 73) look equally different from "Kitten" (ID 156), even though cat and kitten are related!</p>
                    <p><strong>The Solution:</strong> Map tokens to dense vectors where similar meanings are nearby.</p>
                </div>
                
                <div class="card-grid">
                    <div>
                        <label for="emb-word1">Word 1:</label>
                        <input type="text" id="emb-word1" class="text-input" value="king">
                    </div>
                    <div>
                        <label for="emb-word2">Word 2:</label>
                        <input type="text" id="emb-word2" class="text-input" value="queen">
                    </div>
                    <div>
                        <label for="emb-dim">Embedding Dimension:</label>
                        <input type="range" id="emb-dim" min="8" max="64" value="32" oninput="document.getElementById('emb-dim-val').innerText=this.value">
                        <span id="emb-dim-val">32</span>
                    </div>
                </div>
                
                <hr>
                <div id="emb-results-container"></div>
                
                <div class="alert alert-tip" style="margin-top: 20px;">
                    💡 <strong>Note:</strong> In real systems, embeddings are <em>learned</em> from data, so similar words naturally end up with similar vectors. Our demo uses random vectors seeded by word hash—real embeddings would show king≈queen similarity!
                </div>
            `;
        } else {
            // Visual Domain
            const scenes = Object.keys(SAMPLE_SCENES).map(k => `<option value="${k}">${k}</option>`).join("");
            
            content += `
                <div class="alert alert-info">
                    <p><strong>> 🛸 Zara can identify shape parts now, but Part-7 ("curve") and Part-3 ("sharp_corner") look equally random. She needs a map where similar shapes live nearby.</strong></p>
                    <p><strong>The Problem:</strong> In shape world, a gentle curve and a sharp corner look completely unrelated if we only look at their independent IDs.</p>
                    <p><strong>The Solution:</strong> Map shapes to points on a 2D map so that similar shapes cluster together. The direction and distance between shapes gives them meaning!</p>
                    <p style="margin-top:8px; color: var(--accent-3);"><strong>🔌 Layout analogy:</strong> Proximity in embedding space is like matched transistor placement — you place a differential pair close together to share a thermal gradient. Similar layouts get similar vectors. Try comparing <em>diff_pair</em> and <em>current_mirror</em>!</p>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label for="emb-scene-sel">Select a scene to locate on the map:</label>
                    <select id="emb-scene-sel" class="select-input">
                        ${scenes}
                    </select>
                </div>
                
                <div class="card-grid">
                    <div id="emb-visual-scene" class="workshop-card"></div>
                    <div class="workshop-card" id="emb-visual-results"></div>
                </div>
            `;
        }
        
        content += `
            ${chapterNav("embeddings")}
        </div>`;
        return content;
    },

    attachEvents() {
        const isVisual = AppState.learningPath === "Visual";
        
        if (!isVisual) {
            const w1Input = document.getElementById("emb-word1");
            const w2Input = document.getElementById("emb-word2");
            const dimInput = document.getElementById("emb-dim");
            
            const updateWordEmbeddings = () => {
                this.runTextEmbeddings(w1Input.value, w2Input.value, parseInt(dimInput.value, 10));
            };
            
            w1Input.addEventListener("input", updateWordEmbeddings);
            w2Input.addEventListener("input", updateWordEmbeddings);
            dimInput.addEventListener("change", updateWordEmbeddings);
            
            updateWordEmbeddings();
        } else {
            const sceneSel = document.getElementById("emb-scene-sel");
            
            const updateVisualEmbeddings = () => {
                this.runVisualEmbeddings(sceneSel.value);
            };
            
            sceneSel.addEventListener("change", updateVisualEmbeddings);
            updateVisualEmbeddings();
        }
    },

    runTextEmbeddings(word1, word2, dimensions) {
        const container = document.getElementById("emb-results-container");
        if (!window.TextEmbedding) return;

        const embedder = new TextEmbedding(dimensions);
        const emb1 = embedder.getEmbedding(word1);
        const emb2 = embedder.getEmbedding(word2);
        const safeWord1 = escapeHtml(word1);
        const safeWord2 = escapeHtml(word2);
        
        const sim = embedder.cosineSimilarity(emb1, emb2);

        let html = `
            <div class="metric-row">
                <div class="metric-card" style="margin: 0 auto; max-width: 400px;">
                    <h4>Cosine Similarity</h4>
                    <h2>${sim.toFixed(4)}</h2>
                </div>
            </div>
            
            <div class="card-grid">
                <div class="output-box">
                    <strong>${safeWord1}</strong> embedding (first 8 dims):<br>
                    <code>[${emb1.slice(0, 8).map(v => v.toFixed(3)).join(", ")}...]</code>
                </div>
                <div class="output-box">
                    <strong>${safeWord2}</strong> embedding (first 8 dims):<br>
                    <code>[${emb2.slice(0, 8).map(v => v.toFixed(3)).join(", ")}...]</code>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
    },

    runVisualEmbeddings(sceneName) {
        const scene = SAMPLE_SCENES[sceneName];
        if (!scene || !AppState.engines.embedding) return;

        // Render input scene
        const sceneContainer = document.getElementById("emb-visual-scene");
        sceneContainer.innerHTML = TextUtils.renderSceneHtml(scene, 30, 250);

        // Get embedding
        const embEngine = AppState.engines.embedding;
        const vec = embEngine.get_embedding(scene);

        // Dummy neighbor search
        let dists = [];
        for (let i = 0; i < embEngine.scene_names.length; i++) {
            const nName = embEngine.scene_names[i];
            const v = embEngine.embeddings[i];
            if (nName !== sceneName) {
                const cosSim = TextUtils.cosSim(vec, v);
                dists.push({ name: nName, sim: cosSim });
            }
        }
        
        dists.sort((a, b) => b.sim - a.sim);

        let html = `
            <h3>Vector Distance</h3>
            <p><strong>(Simplified 16D Vector representation)</strong></p>
            <div class="output-box" style="margin-bottom: 20px;">
                <code>[${vec.slice(0, 8).map(v => v.toFixed(3)).join(", ")}...]</code>
            </div>
            
            <p><strong>Nearest Neighbors in Meaning Space:</strong></p>
            <ul>
        `;
        
        for (let i = 0; i < Math.min(3, dists.length); i++) {
            html += `<li><strong>${dists[i].name}</strong> (Sim: ${dists[i].sim.toFixed(2)})</li>`;
        }
        
        html += `</ul>`;
        
        document.getElementById("emb-visual-results").innerHTML = html;
    }
};
