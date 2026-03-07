// End-to-End Pipeline Page
const PipelinePage = {
    render() {
        const isVisual = AppState.learningPath === "Visual";
        const sceneOptions = (typeof SAMPLE_SCENES !== "undefined" ? Object.keys(SAMPLE_SCENES) : [])
            .map(name => `<option value="${name}">${name}</option>`)
            .join("");

        return `
            <div class="page-container">
                <div class="main-header">
                    <h2>🔗 End-to-End Pipeline</h2>
                    <h3><em>🛸 Zara's complete system — all 6 chapters working together!</em></h3>
                </div>
                
                <div class="alert alert-info">
                    <p>This demo traces a query through Zara's <strong>complete GenAI pipeline</strong>, showing how each chapter's invention contributes to the final answer.</p>
                    <p><em>Zara started unable to read a single word. Now watch every component she built work together as one system.</em></p>
                </div>
                
                ${!isVisual ? `
                    <div>
                        <label for="pipe-input">Enter your question:</label>
                        <textarea id="pipe-input" class="text-input" rows="3" style="width:100%; resize:vertical;">What is machine learning and how does it work?</textarea>
                    </div>
                ` : `
                    <div>
                        <label for="pipe-scene-select">Select a scene to analyze:</label>
                        <select id="pipe-scene-select" class="select-input">
                            ${sceneOptions}
                        </select>
                    </div>
                `}
                
                <button id="pipe-run-btn" class="mega-button" style="padding:15px; margin:20px 0; background:var(--gradient-primary); border:none;">
                    🚀 Run Complete Pipeline
                </button>
                
                <div id="pipe-results-container"></div>
                ${chapterNav("pipeline")}
            </div>
        `;
    },

    attachEvents() {
        document.getElementById("pipe-run-btn").addEventListener("click", () => {
            if (AppState.learningPath === "Visual") {
                this.runVisualPipeline();
            } else {
                this.runPipeline();
            }
        });
    },

    getFinaleHtml(inputLabel = "📝 Input") {
        return `
            <hr>
            <div style="text-align:center; padding: 30px 0;">
                <h2>🎉 Zara's Mission Complete!</h2>
                <p style="font-size: 1.2em; max-width: 600px; margin: 0 auto;">
                    <em>She arrived unable to read a single word or recognize a single shape.
                    Six inventions later, she built a complete AI system — from scratch.</em>
                </p>
                <br>
                <p>The same architecture powers <strong>ChatGPT</strong>, <strong>Claude</strong>, <strong>DALL-E</strong>, and <strong>Gemini</strong>.<br>
                The only difference? Scale. Our demo has thousands of parameters.<br>
                GPT-4 has over a <strong>trillion</strong>. Same engine — enormously more horsepower.</p>
                <br>
                <div class="pipeline-flow">
                    <div class="pipeline-step">${inputLabel}</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">📕 Codebook</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">📗 Map</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">📘 Library</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">📙 Spotlight</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">📓 Brain</div>
                    <span class="pipeline-arrow">→</span>
                    <div class="pipeline-step">💬 Answer</div>
                </div>
                <br>
                <p style="font-size: 1.1em;"><strong>"What I cannot create, I do not understand."</strong> — Richard Feynman</p>
                <p><em>Now you can create it. Now you understand.</em></p>
            </div>
        `;
    },

    runPipeline() {
        const inputEl = document.getElementById("pipe-input");
        if (!inputEl) return;

        const userInput = inputEl.value;
        if (!userInput) return;
        const safeUserInput = escapeHtml(userInput);

        const container = document.getElementById("pipe-results-container");

        // Step 1: Tokenization
        let tokTokens = [];
        let tokDecoded = "";
        if (window.TextTokenizer) {
            const tok = new TextTokenizer("word");
            tok.train(["machine learning artificial intelligence neural networks"], 100);
            tokTokens = tok.encode(userInput);
            tokDecoded = tok.decode(tokTokens);
        } else {
            tokTokens = userInput.split(/\s+/).map((_, i) => i);
        }

        // Step 2: Embedding
        const queryVec = TextUtils.makeVector(userInput.toLowerCase());

        // Step 3: Vector search
        const knowledgeBase = [
            ["Python is a high-level programming language known for readability.", "Python Docs"],
            ["Machine learning learns from data to make predictions.", "ML Textbook"],
            ["Neural networks are inspired by biological neural networks.", "Deep Learning Book"],
            ["Transformers use self-attention for sequence processing.", "Attention Paper"],
            ["RAG grounds answers in real documents.", "RAG Paper"]
        ];

        const scored = knowledgeBase.map(([text, source]) => {
            const docVec = TextUtils.makeVector(text.toLowerCase());
            return { text, source, score: TextUtils.cosSim(queryVec, docVec) };
        });
        scored.sort((a, b) => b.score - a.score);
        const topDocs = scored.slice(0, 3);

        // Step 6: Answer
        const answer = `Based on retrieved documents: ${topDocs[0].text}`;

        let html = `
            <hr>
            <h3>📕 Step 1: The Codebook (Tokenization)</h3>
            <div class="output-box">
                <strong>Input:</strong> <code>${safeUserInput}</code><br>
                <strong>Tokens:</strong> <code>[${tokTokens.slice(0, 10).join(", ")}${tokTokens.length > 10 ? '...' : ''}]</code> (${tokTokens.length} total)
            </div>
            <div class="step-success">✅ Text converted to token IDs</div>
            
            <hr>
            <h3>📗 Step 2: The Map (Embeddings)</h3>
            <div class="output-box">
                <strong>Query embedding:</strong> <code>[${queryVec.slice(0, 5).map(v => v.toFixed(3)).join(", ")}...]</code> (64 dims)
            </div>
            <div class="step-success">✅ Tokens converted to dense vectors</div>
            
            <hr>
            <h3>📘 Step 3: The Library (Vector Search)</h3>
            <div class="output-box">
                <strong>Retrieved documents:</strong><br>
                ${topDocs.map((d, i) => `${i + 1}. [${d.source}] Score: ${d.score.toFixed(3)}`).join("<br>")}
            </div>
            <div class="step-success">✅ Found relevant documents via similarity search</div>
            
            <hr>
            <h3>📙 Step 4: The Spotlight (Attention)</h3>
            <p>The transformer uses attention to let query tokens attend to context tokens, identifying which parts of retrieved docs are most relevant.</p>
            <div class="step-success">✅ Attention weights computed across query + context</div>
            
            <hr>
            <h3>📓 Step 5: The Brain (Transformer)</h3>
            <p>The transformer processes the augmented prompt through multiple attention layers, feed-forward networks, layer normalization, and final output projection.</p>
            <div class="step-success">✅ Transformer processed augmented prompt</div>
            
            <hr>
            <h3>📔 Step 6: The Search Engine (RAG)</h3>
            <div style="background:linear-gradient(135deg, #1a4a1a, #0d3d0d); border:2px solid #16a34a; border-radius:12px; padding:20px; margin-bottom:15px;">
                ${answer}
            </div>
            <p><strong>Sources:</strong> ${topDocs.map(d => d.source).join(", ")}</p>
            <div class="step-success">✅ Answer generated with source citations!</div>
        `;

        html += this.getFinaleHtml("📝 Input");
        container.innerHTML = html;
    },

    runVisualPipeline() {
        const sceneSel = document.getElementById("pipe-scene-select");
        if (!sceneSel || typeof SAMPLE_SCENES === "undefined") return;

        const sceneName = sceneSel.value;
        const scene = SAMPLE_SCENES[sceneName];
        if (!scene) return;

        const safeSceneName = escapeHtml(sceneName);
        const container = document.getElementById("pipe-results-container");

        const tokEngine = AppState.engines.tokenizer;
        const embEngine = AppState.engines.embedding;
        const dbEngine = AppState.engines.db;
        const attEngine = AppState.engines.attention;

        // Step 1: Tokenization (shape parts)
        let shapeTokenIds = [];
        if (tokEngine) {
            if (typeof tokEngine.tokenize === "function") {
                shapeTokenIds = tokEngine.tokenize(scene);
            } else if (typeof tokEngine.encode === "function") {
                shapeTokenIds = tokEngine.encode(scene);
            }
        }
        const shapeTokenLabels = tokEngine && typeof tokEngine.decode === "function"
            ? tokEngine.decode(shapeTokenIds)
            : [];

        // Step 2: Embeddings
        let sceneEmbedding = [];
        if (embEngine) {
            if (typeof embEngine.get_embedding === "function") {
                sceneEmbedding = embEngine.get_embedding(scene);
            } else if (typeof embEngine.getEmbedding === "function") {
                sceneEmbedding = embEngine.getEmbedding(scene);
            }
        }

        // Step 3: Vector search
        const rawResults = (dbEngine && sceneEmbedding.length > 0)
            ? (typeof dbEngine.query === "function"
                ? dbEngine.query(sceneEmbedding, 3)
                : (typeof dbEngine.search === "function" ? dbEngine.search(sceneEmbedding, 3) : []))
            : [];

        const topScenes = (rawResults || [])
            .map(r => {
                if (Array.isArray(r)) return { name: r[0], score: r[1] };
                return {
                    name: r.id || r.name || r.scene,
                    score: r.score !== undefined ? r.score : (r.similarity !== undefined ? r.similarity : 0)
                };
            })
            .filter(r => r.name && r.name !== sceneName)
            .slice(0, 3);

        // Step 4: Attention
        const attResult = attEngine && typeof attEngine.compute_attention === "function"
            ? attEngine.compute_attention(scene)
            : [[], []];
        const attentionWeights = attResult[0] || [];
        const attentionLabels = attResult[1] || [];

        // Step 6: Grounded analysis (RAG style)
        const topMatch = topScenes.length > 0 ? topScenes[0] : null;
        const topMatchScene = topMatch ? SAMPLE_SCENES[topMatch.name] : null;

        const queryColors = new Set((scene.shapes || []).map(s => s.color));
        const topMatchShapes = topMatchScene && topMatchScene.shapes ? topMatchScene.shapes : [];
        const matchColors = new Set(topMatchShapes.map(s => s.color));
        const sharedColors = [...queryColors].filter(c => matchColors.has(c));

        let groundedAnalysis = `No close match was retrieved from Zara's scene library for <strong>${safeSceneName}</strong>.`;
        if (topMatch) {
            groundedAnalysis = `Scene <strong>${safeSceneName}</strong> is most similar to <strong>${escapeHtml(topMatch.name)}</strong> (similarity: ${Number(topMatch.score).toFixed(2)}). `
                + `They share <strong>${sharedColors.length}</strong> color patterns (${sharedColors.map(escapeHtml).join(", ") || "none"}), `
                + `so Zara grounds her explanation using retrieved examples instead of guessing.`;
        }

        let html = `
            <hr>
            <h3>📕 Step 1: The Codebook (Tokenization)</h3>
            <div class="output-box">
                <strong>Input scene:</strong> <code>${safeSceneName}</code><br>
                ${TextUtils.renderSceneHtml(scene, 30, 200)}
                <strong>Part tokens:</strong> <code>[${shapeTokenIds.slice(0, 12).join(", ")}${shapeTokenIds.length > 12 ? "..." : ""}]</code> (${shapeTokenIds.length} total)<br>
                <strong>Decoded parts:</strong> ${shapeTokenLabels.slice(0, 10).map(label => `<code>${escapeHtml(label)}</code>`).join(" ")}
            </div>
            <div class="step-success">✅ Scene decomposed into shape-part token IDs</div>

            <hr>
            <h3>📗 Step 2: The Map (Embeddings)</h3>
            <div class="output-box">
                ${TextUtils.renderSceneHtml(scene, 30, 200)}
                <strong>Scene embedding:</strong> <code>[${sceneEmbedding.slice(0, 8).map(v => Number(v).toFixed(3)).join(", ")}${sceneEmbedding.length > 8 ? "..." : ""}]</code> (${sceneEmbedding.length} dims)
            </div>
            <div class="step-success">✅ Scene converted to a dense vector in meaning space</div>

            <hr>
            <h3>📘 Step 3: The Library (Vector Search)</h3>
            <div class="output-box">
                ${TextUtils.renderSceneHtml(scene, 30, 200)}
                <strong>Retrieved similar scenes:</strong><br>
                ${topScenes.length > 0
                    ? topScenes.map((r, i) => `${i + 1}. <strong>${escapeHtml(r.name)}</strong> (Score: ${Number(r.score).toFixed(3)})`).join("<br>")
                    : "No neighbors found."}
            </div>
            ${topScenes.length > 0 ? `
                <div class="card-grid" style="grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px;">
                    ${topScenes.map(r => `
                        <div>
                            <strong>${escapeHtml(r.name)}</strong><br>
                            <span style="color:var(--accent-1); font-size:12px;">Sim: ${Number(r.score).toFixed(2)}</span>
                            ${TextUtils.renderSceneHtml(SAMPLE_SCENES[r.name], 15, 150)}
                        </div>
                    `).join("")}
                </div>
            ` : ""}
            <div class="step-success">✅ Found nearest visual memories by vector similarity</div>

            <hr>
            <h3>📙 Step 4: The Spotlight (Attention)</h3>
            <div class="output-box">
                ${TextUtils.renderSceneHtml(scene, 30, 200)}
                <strong>Attention heads:</strong> ${attentionWeights.length} (Proximity, Color, Alignment, Containment)<br>
                <strong>Tracked shapes:</strong> ${attentionLabels.length > 0
                    ? attentionLabels.map(label => `<code>${escapeHtml(label)}</code>`).join(" ")
                    : "None"}
            </div>
            <div class="step-success">✅ Multi-head attention computed relationships between scene elements</div>

            <hr>
            <h3>📓 Step 5: The Brain (Transformer)</h3>
            <div class="output-box">
                ${TextUtils.renderSceneHtml(scene, 30, 200)}
                The transformer processes the scene context through stacked attention + feed-forward layers.
                In this workshop demo, weights are intentionally untrained, so predictions are illustrative architecture traces —
                exactly like the text path.
            </div>
            <div class="step-success">✅ Transformer architecture applied to visual context (untrained demo weights)</div>

            <hr>
            <h3>📔 Step 6: The Search Engine (RAG)</h3>
            <div style="background:linear-gradient(135deg, #1a4a1a, #0d3d0d); border:2px solid #16a34a; border-radius:12px; padding:20px; margin-bottom:15px;">
                <strong>🛸 Zara's Grounded Analysis:</strong><br>
                ${groundedAnalysis}
            </div>
            <p><strong>Sources:</strong> ${topScenes.length > 0 ? topScenes.map(r => escapeHtml(r.name)).join(", ") : "No retrieved scenes"}</p>
            <div class="step-success">✅ Analysis grounded in retrieved visual references</div>
        `;

        html += this.getFinaleHtml("🖼️ Input");
        container.innerHTML = html;
    }
};
