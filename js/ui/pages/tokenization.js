// Workshop 1: Tokenization
const TokenizationPage = {
    render() {
        const isVisual = AppState.learningPath === "Visual";
        
        let content = `
            <div class="page-container">
                <div class="main-header">
                    <h2>🔤 Workshop 1: Tokenization</h2>
                    <h3><em>📕 Chapter 1 of Zara's Journey: The Codebook</em></h3>
                </div>
        `;

        if (!isVisual) {
            content += `
                <div class="alert alert-info">
                    <p><strong>> 🛸 Zara can't read human text. She needs a codebook to turn symbols into numbers.</strong></p>
                    <p><strong>The Problem:</strong> Computers don't understand text—only numbers!</p>
                    <p><strong>The Solution:</strong> Break text into tokens and assign each a number.</p>
                </div>
                
                <div class="card-grid">
                    <div>
                        <label for="tok-input">Enter text to tokenize:</label>
                        <input type="text" id="tok-input" class="text-input" value="The quick brown fox jumps over the lazy dog.">
                    </div>
                    <div>
                        <label for="tok-strategy">Strategy:</label>
                        <select id="tok-strategy" class="select-input">
                            <option value="char">Character (char)</option>
                            <option value="word" selected>Word (word)</option>
                            <option value="bpe">Byte-Pair Encoding (bpe snippet)</option>
                        </select>
                    </div>
                </div>
                
                <hr>
                
                <div id="tok-results-container">
                    <!-- Results injected here -->
                </div>
            `;
        } else {
            // Visual Domain
            // Get available scenes from global shape data
            const scenes = Object.keys(SAMPLE_SCENES).map(k => `<option value="${k}">${k}</option>`).join("");
            
            content += `
                <div class="alert alert-info">
                    <p><strong>Visual Domain:</strong> Watch Zara break down an image into sub-shape primitives.</p>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <label for="tok-scene-sel">Choose a scene to parse:</label>
                    <select id="tok-scene-sel" class="select-input">
                        ${scenes}
                    </select>
                </div>
                
                <div class="card-grid">
                    <div id="tok-visual-scene" class="workshop-card">
                        <!-- Scene renders here -->
                    </div>
                    <div class="workshop-card">
                        <h3>Part Tokenization</h3>
                        <div id="tok-visual-stats" class="metric-row"></div>
                        <div id="tok-visual-boxes" style="line-height:2.2; margin-top:20px;"></div>
                    </div>
                </div>
            `;
        }
        
        content += `</div>`;
        return content;
    },

    attachEvents() {
        const isVisual = AppState.learningPath === "Visual";
        
        if (!isVisual) {
            const inputEl = document.getElementById("tok-input");
            const selectEl = document.getElementById("tok-strategy");
            
            const updateTextTok = () => {
                const text = inputEl.value;
                const strategy = selectEl.value;
                this.runTextTokenization(text, strategy);
            };
            
            inputEl.addEventListener("input", updateTextTok);
            selectEl.addEventListener("change", updateTextTok);
            
            // Initial run
            updateTextTok();
        } else {
            const sceneSel = document.getElementById("tok-scene-sel");
            
            const updateVisualTok = () => {
                const sceneName = sceneSel.value;
                this.runVisualTokenization(sceneName);
            };
            
            sceneSel.addEventListener("change", updateVisualTok);
            
            // Initial run
            updateVisualTok();
        }
    },

    runTextTokenization(text, strategy) {
        // We need to implement SimpleTokenizer in JS first (see new src/ml/tokenizer.js)
        const container = document.getElementById("tok-results-container");
        if (!window.TextTokenizer) {
            container.innerHTML = "<p>Loading Tokenizer Engine...</p>";
            return;
        }

        const corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming how we build software.",
            "Python is a popular programming language.",
            "The fox is quick and clever."
        ];
        
        const tokenizer = new TextTokenizer(strategy);
        tokenizer.train(corpus, 100);
        
        const tokens = tokenizer.encode(text);
        const decoded = tokenizer.decode(tokens);
        const compression = Math.max(tokens.length, 1);
        const ratio = (text.length / compression).toFixed(1);

        let html = `
            <div class="metric-row">
                <div class="metric-card">
                    <h4>Vocab Size</h4>
                    <h2>${tokenizer.vocabSize()}</h2>
                </div>
                <div class="metric-card">
                    <h4>Token Count</h4>
                    <h2>${tokens.length}</h2>
                </div>
                <div class="metric-card">
                    <h4>Compression</h4>
                    <h2>${ratio}x</h2>
                </div>
            </div>
            
            <div class="output-box">
                <strong>Token IDs:</strong><br>
                <code>[${tokens.join(", ")}]</code>
            </div>
            
            <div class="output-box">
                <strong>Decoded:</strong><br>
                <code>${decoded}</code>
            </div>
        `;

        if (strategy === "word") {
            const words = decoded.split(" ").filter(w => w.trim() !== "");
            html += `<div class="card-grid" style="grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));">`;
            for(let i=0; i < Math.min(words.length, 8); i++) {
                html += `
                    <div style="text-align:center; padding:15px; background:var(--gradient-primary); border-radius:8px; box-shadow:var(--shadow-soft);">
                        <div style="font-size:1.1rem; font-weight:bold;">${words[i]}</div>
                        <div style="font-size:0.8rem; opacity:0.9;">ID: ${tokens[i]}</div>
                    </div>
                `;
            }
            html += `</div>`;
        }
        
        container.innerHTML = html;
    },

    runVisualTokenization(sceneName) {
        const scene = SAMPLE_SCENES[sceneName];
        if (!scene || !AppState.engines.tokenizer) return;

        // Render Scene HTML
        const sceneContainer = document.getElementById("tok-visual-scene");
        sceneContainer.innerHTML = this.renderSceneHtml(scene);

        // Run Tokenizer
        const tok = AppState.engines.tokenizer;
        const ids = tok.encode(scene);
        const labels = tok.decode(ids);

        const statsContainer = document.getElementById("tok-visual-stats");
        statsContainer.innerHTML = `
            <div class="metric-card">
                <h4>Total Tokens</h4>
                <h2>${ids.length}</h2>
            </div>
        `;

        const boxesContainer = document.getElementById("tok-visual-boxes");
        const boxesHtml = labels.slice(0, 40).map(l => 
            `<span style="display:inline-block; padding:4px 8px; margin:2px; background:#10b981; border-radius:4px; font-family:monospace; font-size:13px; color:white;">${l}</span>`
        ).join("");
        
        boxesContainer.innerHTML = boxesHtml;
    },

    // Helper replicated from app.py
    renderSceneHtml(scene, scale = 30, h = 300) {
        let html = `<div class="shape-overlay" style="height:${h}px; position:relative; background:#0f172a; border-radius:8px; overflow:hidden;">`;
        for (const s of (scene.shapes || [])) {
            const colorHex = LAYER_COLORS[s.color] || "#888888";
            const left = s.x * scale;
            const top = s.y * scale;
            const w = s.w * scale;
            const h_s = s.h * scale;
            html += `
                <div style="position:absolute; left:${left}px; top:${top}px; 
                            width:${w}px; height:${h_s}px; background:${colorHex}80; 
                            border:2px solid ${colorHex}; border-radius:4px; 
                            display:flex; align-items:center; justify-content:center; 
                            font-size:10px; color:white; text-shadow:1px 1px 2px #000;">
                    ${s.label}
                </div>
            `;
        }
        html += `</div>`;
        return html;
    }
};
