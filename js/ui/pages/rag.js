// Workshop 6: RAG
const RAGPage = {
    textKnowledgeBase: [
        ["Python is a high-level, general-purpose programming language known for its readability.", "Python Docs"],
        ["Machine learning is a subset of artificial intelligence that learns from data.", "ML Textbook"],
        ["Neural networks are computing systems inspired by biological neural networks.", "Deep Learning Book"],
        ["JavaScript is a programming language used for web development.", "MDN Web Docs"],
        ["Transformers are a type of neural network architecture using self-attention.", "Attention Paper"],
        ["RAG combines retrieval with generation to ground answers in real documents.", "RAG Paper"],
        ["Vector databases store embeddings and enable fast similarity search.", "VectorDB Guide"],
        ["Tokenization converts text to numerical tokens that models can process.", "NLP Fundamentals"]
    ],

    render() {
        const isVisual = AppState.learningPath === "Visual";

        let content = `
            <div class="page-container">
                <div class="main-header">
                    <h2>🔍 Workshop 6: RAG</h2>
                    <h3><em>📔 Chapter 6 of Zara's Journey: The Connection</em></h3>
                </div>
                
                <div class="alert alert-info">
                    <p><strong>> 🛸 Zara's brain works, but sometimes makes things up. Her solution: look things up BEFORE answering — don't hallucinate, verify!</strong></p>
                    <p><strong>The Problem:</strong> ${isVisual ? "Zara's shape transformer guesses random shapes. It needs to look up REAL scenes before predicting." : "LLMs can hallucinate, have knowledge cutoffs, and can't cite sources."}</p>
                    <p><strong>The Solution:</strong> RAG = Retrieval + Augmented + Generation</p>
                </div>
        `;

        if (!isVisual) {
            content += `
                <p><strong>Knowledge base: ${this.textKnowledgeBase.length} documents</strong></p>
                
                <div class="card-grid">
                    <div>
                        <label for="rag-query">Ask a question:</label>
                        <input type="text" id="rag-query" class="text-input" value="What is Python?" style="width:100%;">
                    </div>
                </div>
                
                <button id="rag-search-btn" class="mega-button" style="padding:15px; margin:20px 0;">🔍 Get Answer</button>
                
                <div id="rag-results-container"></div>
            `;
        } else {
            // Visual Domain
            const scenes = Object.keys(SAMPLE_SCENES).map(k => `<option value="${k}">${k}</option>`).join("");

            content += `
                <p><strong>Shape Knowledge Base: ${Object.keys(SAMPLE_SCENES).length} indexed scenes</strong></p>
                
                <div class="card-grid" style="grid-template-columns: 1fr 2fr;">
                    <div>
                        <label for="rag-scene-sel">Select a query scene:</label>
                        <select id="rag-scene-sel" class="select-input" style="margin-bottom:15px;">
                            ${scenes}
                        </select>
                        <p><strong>Query Scene:</strong></p>
                        <div id="rag-query-scene"></div>
                    </div>
                    <div>
                        <h3>RAG Pipeline Results</h3>
                        <div id="rag-visual-results"></div>
                    </div>
                </div>
            `;
        }

        content += `
            ${chapterNav("rag")}
        </div>`;
        return content;
    },

    attachEvents() {
        const isVisual = AppState.learningPath === "Visual";

        if (!isVisual) {
            document.getElementById("rag-search-btn").addEventListener("click", () => this.runTextRAG());
        } else {
            const sceneSel = document.getElementById("rag-scene-sel");
            sceneSel.addEventListener("change", () => this.runVisualRAG(sceneSel.value));
            this.runVisualRAG(sceneSel.value);
        }
    },

    runTextRAG() {
        const query = document.getElementById("rag-query").value;
        if (!query) return;

        const qVec = TextUtils.makeVector(query.toLowerCase());

        const scored = this.textKnowledgeBase.map(([text, source]) => {
            const docVec = TextUtils.makeVector(text.toLowerCase());
            return { text, source, score: TextUtils.cosSim(qVec, docVec) };
        });

        scored.sort((a, b) => b.score - a.score);
        const topDocs = scored.slice(0, 3);

        const answer = `Based on retrieved documents: ${topDocs[0].text}`;
        const sources = topDocs.map(d => d.source);

        let html = `
            <h3>💬 Answer</h3>
            <div style="background: linear-gradient(135deg, #1a4a1a, #0d3d0d);
                        border: 2px solid #16a34a; border-radius: 12px; padding: 20px; margin-bottom:20px;">
                ${answer}
            </div>
            
            <h3>📚 Sources</h3>
            <div class="alert alert-tip">${sources.join(", ")}</div>
            
            <h3>📄 Retrieved Documents</h3>
        `;

        topDocs.forEach(doc => {
            html += `
                <div class="workshop-card" style="margin-bottom:10px;">
                    <strong>Score: ${doc.score.toFixed(3)}</strong> | Source: ${doc.source}<br>
                    <span style="color:var(--text-secondary);">${doc.text}</span>
                </div>
            `;
        });

        document.getElementById("rag-results-container").innerHTML = html;
    },

    runVisualRAG(sceneName) {
        const scene = SAMPLE_SCENES[sceneName];
        if (!scene || !AppState.engines.embedding || !AppState.engines.db) return;

        // Render query scene
        document.getElementById("rag-query-scene").innerHTML = TextUtils.renderSceneHtml(scene, 30, 200);

        // Step 1: Embed & Retrieve
        const qEmb = AppState.engines.embedding.get_embedding(scene);
        const results = AppState.engines.db.search(qEmb, 4);
        const filtered = results.filter(r => r[0] !== sceneName).slice(0, 3);

        // Step 2: Use the retrieved scenes to "generate" an answer
        const topScene = filtered.length > 0 ? SAMPLE_SCENES[filtered[0][0]] : null;

        let html = `
            <div class="step-success">📗 Step 1: Embedded query scene → ${qEmb.length}D vector</div>
            <div class="step-success">📘 Step 2: Retrieved ${filtered.length} similar scenes from the database</div>
        `;

        // Show retrieved scenes
        html += `<div class="card-grid" style="grid-template-columns: repeat(3, 1fr); gap:10px; margin:15px 0;">`;
        filtered.forEach(res => {
            const rName = res[0];
            const rScore = res[1];
            const rScene = SAMPLE_SCENES[rName];
            html += `
                <div style="text-align:center;">
                    <strong>${rName}</strong><br>
                    <span style="color:var(--accent-1); font-size:0.85rem;">Sim: ${rScore.toFixed(2)}</span>
                    ${TextUtils.renderSceneHtml(rScene, 15, 120)}
                </div>
            `;
        });
        html += `</div>`;

        html += `<div class="step-success">📙 Step 3: Attended to retrieved context</div>`;

        // Generate answer
        if (topScene) {
            const sharedColors = {};
            (scene.shapes || []).forEach(s => sharedColors[s.color] = true);
            const matchingShapes = (topScene.shapes || []).filter(s => sharedColors[s.color]);

            html += `
                <div class="step-success">📔 Step 4: Generated grounded answer</div>
                
                <div style="background: linear-gradient(135deg, #1a4a1a, #0d3d0d);
                            border: 2px solid #16a34a; border-radius: 12px; padding: 20px; margin-top:15px;">
                    <strong>🛸 Zara's Grounded Analysis:</strong><br>
                    "The scene <strong>${sceneName}</strong> is most similar to <strong>${filtered[0][0]}</strong> 
                    (similarity: ${filtered[0][1].toFixed(2)}). 
                    They share ${matchingShapes.length} shapes with matching colors. 
                    Based on retrieved context, this scene likely represents a 
                    <strong>${scene.name || sceneName}</strong> composition."
                </div>
                
                <div class="alert alert-tip" style="margin-top:15px;">
                    📚 <strong>Sources:</strong> ${filtered.map(r => r[0]).join(", ")} (from Shape Knowledge Base)
                </div>
            `;
        }

        document.getElementById("rag-visual-results").innerHTML = html;
    }
};
