// Workshop 5: Transformers
const TransformersPage = {
    render() {
        const isVisual = AppState.learningPath === "Visual";

        let content = `
            <div class="page-container">
                <div class="main-header">
                    <h2>🧠 Workshop 5: Transformers</h2>
                    <h3><em>📓 Chapter 5 of Zara's Journey: The Brain</em></h3>
                </div>
                
                <div class="alert alert-info">
                    <p><strong>> 🛸 Zara has all the pieces — a codebook, a meaning map, attention spotlights. Now she assembles them into a working brain. She calls it... a Transformer.</strong></p>
                    <p><strong>The Architecture:</strong> Transformers combine Embeddings (2) + Attention (4) + Feed-forward layers + Stacking for deeper understanding.</p>
                </div>
                
                <div class="output-box" style="text-align:center; font-size:1.1rem; padding:20px;">
                    Input → Embedding → [Attention → FFN] × N → Output<br>
                    <span style="color:var(--accent-1);">↑__________________| (residual connections)</span>
                </div>
                
                <hr>
        `;

        if (!isVisual) {
            content += `
                <h3>🎮 Try Generation (Text Domain)</h3>
                
                <div class="card-grid">
                    <div>
                        <label for="tf-prompt">Start of sentence:</label>
                        <input type="text" id="tf-prompt" class="text-input" value="The cat">
                    </div>
                    <div>
                        <label for="tf-tokens">Tokens to generate:</label>
                        <input type="range" id="tf-tokens" min="1" max="10" value="5"
                               oninput="document.getElementById('tf-tokens-val').innerText=this.value">
                        <span id="tf-tokens-val">5</span>
                    </div>
                </div>
                
                <button id="tf-gen-btn" class="mega-button" style="padding:15px; margin:20px 0;">🧠 Generate</button>
                
                <div id="tf-results-container"></div>
                
                <div class="alert alert-warning" style="margin-top:20px;">
                    ⚠️ <strong>Note:</strong> This transformer has random (untrained) weights, so it produces random output. Real transformers are trained on billions of words to learn meaningful patterns!
                </div>
            `;
        } else {
            // Visual Domain
            const scenes = Object.keys(SAMPLE_SCENES).map(k => `<option value="${k}">${k}</option>`).join("");

            content += `
                <h3>🛸 Visual Domain: Shape-to-Shape Transformer</h3>
                <p>Zara's transformer takes a scene, runs it through the entire pipeline (Tokenize → Embed → Attend → Transform), and tries to predict the <strong>next shape</strong> to add to the scene.</p>
                
                <div class="card-grid" style="grid-template-columns: 1fr 1fr;">
                    <div>
                        <label for="tf-scene-sel">Select input scene:</label>
                        <select id="tf-scene-sel" class="select-input" style="margin-bottom:15px;">
                            ${scenes}
                        </select>
                        <div id="tf-scene-visual"></div>
                    </div>
                    <div class="workshop-card" id="tf-visual-results">
                        <!-- Populated by JS -->
                    </div>
                </div>
                
                <div class="alert alert-warning" style="margin-top:20px;">
                    ⚠️ <strong>Note:</strong> Just like the text path, our shape transformer has untrained weights — it picks random shapes. A real visual transformer (like DALL-E) is trained on billions of images to learn meaningful shape predictions. <strong>The architecture is identical — scale is the difference.</strong>
                </div>
            `;
        }

        content += `
            ${chapterNav("transformers")}
        </div>`;
        return content;
    },

    attachEvents() {
        const isVisual = AppState.learningPath === "Visual";

        if (!isVisual) {
            document.getElementById("tf-gen-btn").addEventListener("click", () => this.runGeneration());
        } else {
            const sceneSel = document.getElementById("tf-scene-sel");
            sceneSel.addEventListener("change", () => this.runVisualGeneration(sceneSel.value));
            this.runVisualGeneration(sceneSel.value);
        }
    },

    runGeneration() {
        const prompt = document.getElementById("tf-prompt").value;
        const maxTokens = parseInt(document.getElementById("tf-tokens").value, 10);
        const words = prompt.toLowerCase().split(/\s+/).filter(w => w);

        const wordToId = {};
        const uniqueWords = [...new Set(words)];
        uniqueWords.forEach((w, i) => wordToId[w] = i + 1);

        const inputIds = words.map(w => wordToId[w] || 0);

        const vocabSize = 100;
        const generatedIds = [...inputIds];
        
        let seed = 42;
        for (let i = 0; i < maxTokens; i++) {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            generatedIds.push(seed % vocabSize);
        }

        const newTokens = generatedIds.slice(inputIds.length);

        document.getElementById("tf-results-container").innerHTML = `
            <div class="output-box">
                <strong>Input token IDs:</strong><br>
                <code>[${inputIds.join(", ")}]</code>
            </div>
            <div class="output-box">
                <strong>Generated token IDs:</strong><br>
                <code>[${newTokens.join(", ")}]</code>
            </div>
            <p style="color:var(--text-secondary); margin-top:15px;">
                These are random IDs because our transformer has untrained weights.
                Real transformers like GPT-4 have over a <strong>trillion</strong> parameters
                trained on billions of words — the magic isn't the architecture, it's the scale.
            </p>
        `;
    },

    runVisualGeneration(sceneName) {
        const scene = SAMPLE_SCENES[sceneName];
        if (!scene) return;

        // Render the input scene
        document.getElementById("tf-scene-visual").innerHTML = TextUtils.renderSceneHtml(scene, 30, 250);

        // Run through the pipeline
        const tok = AppState.engines.tokenizer;
        const emb = AppState.engines.embedding;
        const att = AppState.engines.attention;

        const tokenIds = tok ? tok.encode(scene) : [];
        const embedding = emb ? emb.get_embedding(scene) : [];
        const attResult = att ? att.compute_attention(scene) : [[], []];
        const attWeights = attResult[0];
        const attLabels = attResult[1];

        // "Predict" next shapes (random, like the text domain)
        const shapeTypes = Object.keys(SHAPE_DECOMPOSITION);
        const colorNames = Object.keys(LAYER_COLORS);
        let seed = 42 + sceneName.length;
        const predictions = [];
        for (let i = 0; i < 3; i++) {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            const type = shapeTypes[seed % shapeTypes.length];
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            const color = colorNames[seed % colorNames.length];
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            const conf = (30 + (seed % 40)) / 100;
            predictions.push({ type, color, confidence: conf });
        }

        let html = `
            <h3>Transformer Pipeline</h3>
            
            <div class="step-success">📕 Step 1: Tokenized → ${tokenIds.length} part tokens</div>
            
            <div class="step-success">📗 Step 2: Embedded → ${embedding.length}D vector</div>
            <div class="output-box" style="font-size:0.85rem;">
                <code>[${embedding.slice(0, 6).map(v => v.toFixed(3)).join(", ")}...]</code>
            </div>
            
            <div class="step-success">📙 Step 3: Attention → ${attLabels.length} shapes attended</div>
            
            <div class="step-success">📓 Step 4: Transform → Predicted next shapes:</div>
        `;

        predictions.forEach((p, i) => {
            const colorHex = LAYER_COLORS[p.color] || "#888";
            html += `
                <div style="display:flex; align-items:center; gap:12px; margin:8px 0; padding:10px; background:var(--bg-primary); border-radius:8px; border:1px solid var(--border-color);">
                    <div style="width:40px; height:40px; background:${colorHex}80; border:2px solid ${colorHex}; border-radius:6px;"></div>
                    <div>
                        <strong>${p.type}</strong> (${p.color})<br>
                        <span style="color:var(--text-secondary); font-size:0.85rem;">Confidence: ${(p.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>
            `;
        });

        document.getElementById("tf-visual-results").innerHTML = html;
    }
};
