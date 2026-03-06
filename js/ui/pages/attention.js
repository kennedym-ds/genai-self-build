// Workshop 4: Attention
const AttentionPage = {
    render() {
        const isVisual = AppState.learningPath === "Visual";
        
        let content = `
            <div class="page-container">
                <div class="main-header">
                    <h2>👀 Workshop 4: Attention</h2>
                    <h3><em>📙 Chapter 4 of Zara's Journey: The Breakthrough</em></h3>
                </div>
        `;

        if (!isVisual) {
            content += `
                <div class="alert alert-info">
                    <p><strong>> 🛸 "The bank was near the river bank" — which bank is which? Zara invents a spotlight system where every word asks: "Who is relevant to me?"</strong></p>
                    <p><strong>The Problem:</strong> In "The cat sat on the mat because it was tired", what does "it" refer to?</p>
                    <p><strong>The Solution:</strong> Attention lets each word look at all other words and decide what's relevant context.</p>
                </div>
                
                <div class="card-grid">
                    <div>
                        <label for="att-input">Enter a sentence:</label>
                        <input type="text" id="att-input" class="text-input" value="The cat sat on the mat" style="width: 100%;">
                    </div>
                </div>
                
                <hr>
                
                <div id="att-results-container"></div>
                
                <div class="alert alert-tip" style="margin-top: 20px;">
                    💡 <strong>Reading the matrix:</strong> Each row shows attention distribution for one word. Higher values (darker blue) mean that word pays more attention to that column's word.
                </div>
            `;
        } else {
            // Visual Domain
            const scenes = Object.keys(SAMPLE_SCENES).map(k => `<option value="${k}">${k}</option>`).join("");
            
            content += `
                <div class="alert alert-info">
                    <p><strong>The Problem:</strong> A red circle could be a traffic light, or it could be a clown's nose. Context determines meaning, even for shapes.</p>
                    <p><strong>The Solution:</strong> Zara sets up four "spotlights" (Attention Heads) to check relationships: Proximity, Color, Alignment, and Containment. Shapes "pay attention" to other shapes that are relevant to them.</p>
                </div>
                
                <div class="card-grid" style="grid-template-columns: 1fr 2fr;">
                    <div>
                        <label for="att-scene-sel">Select a scene to analyze:</label>
                        <select id="att-scene-sel" class="select-input" style="margin-bottom: 20px;">
                            ${scenes}
                        </select>
                        <div id="att-visual-scene"></div>
                    </div>
                    
                    <div class="workshop-card">
                        <h3>Multi-Head Attention</h3>
                        <div style="margin-bottom: 15px;" id="att-head-controls">
                            <label><input type="radio" name="att_head" value="0" checked> 1. Proximity</label>&nbsp;
                            <label><input type="radio" name="att_head" value="1"> 2. Color Match</label>&nbsp;
                            <label><input type="radio" name="att_head" value="2"> 3. Alignment</label>&nbsp;
                            <label><input type="radio" name="att_head" value="3"> 4. Containment</label>
                        </div>
                        <div id="att-visual-results"></div>
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
            const inputEl = document.getElementById("att-input");
            const updateTextAtt = () => this.runTextAttention(inputEl.value);
            inputEl.addEventListener("input", updateTextAtt);
            updateTextAtt();
        } else {
            const sceneSel = document.getElementById("att-scene-sel");
            const radios = document.querySelectorAll('input[name="att_head"]');
            
            const updateVisualAtt = () => {
                const checkedRadio = document.querySelector('input[name="att_head"]:checked');
                this.runVisualAttention(sceneSel.value, parseInt(checkedRadio.value, 10));
            };
            
            sceneSel.addEventListener("change", updateVisualAtt);
            radios.forEach(r => r.addEventListener("change", updateVisualAtt));
            
            updateVisualAtt();
        }
    },

    runTextAttention(sentence) {
        if (!window.TextAttention) return;
        
        const words = sentence.split(" ").filter(w => w.trim() !== "");
        const seqLen = words.length;
        
        if (seqLen === 0) {
            document.getElementById("att-results-container").innerHTML = "";
            return;
        }
        
        const attn = new TextAttention(64, 1);
        
        // Mock embeddings (randomized based on text length for visual stability)
        const x = [];
        for (let i = 0; i < seqLen; i++) {
            const wordEmb = [];
            for (let j = 0; j < 64; j++) {
                let seed = i * 64 + j + sentence.length;
                let val = Math.sin(seed * 1000) * 10000;
                wordEmb.push((val - Math.floor(val)) * 2 - 1);
            }
            x.push(wordEmb);
        }
        
        const result = attn.forward(x);
        const weights = result.weights;
        
        let html = `
            <h3>Attention Weights</h3>
            <p><em>Each row shows how much that word attends to other words</em></p>
            <div style="background:var(--bg-secondary); padding:15px; border-radius:8px; overflow-x:auto;">
                <table style="width:100%; border-collapse:collapse; color:white; text-align:center;">
                    <tr>
                        <th style="padding:10px; border-bottom:1px solid #444;">&rarr;<br>&darr;</th>
        `;
        
        // Matrix headers
        for (const w of words) {
            html += `<th style="padding:10px; border-bottom:1px solid #444;">${w}</th>`;
        }
        html += `</tr>`;
        
        // Matrix rows
        for (let i = 0; i < seqLen; i++) {
            html += `<tr><td style="padding:10px; font-weight:bold; border-right:1px solid #444;">${words[i]}</td>`;
            for (let j = 0; j < seqLen; j++) {
                const val = weights[i][j];
                const bgColors = [
                    [240, 249, 255], // lightest blue
                    [224, 242, 254],
                    [186, 230, 253],
                    [125, 211, 252],
                    [56, 189, 248]  // mid blue
                ];
                
                // Scale value to index 0-4
                const idx = Math.min(Math.floor(val * 4.99), 4);
                const color = bgColors[idx] || bgColors[0];
                const textColor = idx > 2 ? "#000" : "#666";
                
                html += `<td style="padding:10px; background:rgb(${color[0]},${color[1]},${color[2]}); color:${textColor}; border:1px solid #fff;">${val.toFixed(3)}</td>`;
            }
            html += `</tr>`;
        }
        
        html += `</table></div>`;
        document.getElementById("att-results-container").innerHTML = html;
    },

    runVisualAttention(sceneName, headIdx) {
        const scene = SAMPLE_SCENES[sceneName];
        if (!scene || !AppState.engines.attention) return;

        // Render input
        document.getElementById("att-visual-scene").innerHTML = TokenizationPage.renderSceneHtml(scene);

        // Compute attention matrix from shapes_core.js
        const result = AppState.engines.attention.compute_attention(scene);
        const allWeights = result[0];
        const labels = result[1];

        if (labels.length === 0) {
            document.getElementById("att-visual-results").innerHTML = "<p>No shapes in scene to attend to.</p>";
            return;
        }

        const headWeights = allWeights[headIdx];

        let html = `
            <div style="background:#1e293b; padding:10px; border-radius:8px; overflow-x:auto;">
                <table style="color:white; font-size:12px; border-collapse:collapse; text-align:center; width:100%;">
                    <tr>
                        <th style="padding:4px; border-bottom:1px solid #444;">&rarr;<br>&darr;</th>
        `;

        for (const l of labels) {
            html += `<th style="padding:4px; border-bottom:1px solid #444; max-width:40px; overflow:hidden; text-overflow:ellipsis;" title="${l}">${l.substring(0, 5)}...</th>`;
        }
        html += `</tr>`;

        for (let i = 0; i < headWeights.length; i++) {
            const rowLabel = labels[i].substring(0, 5) + "...";
            html += `<tr><td style="padding:4px; font-weight:bold; text-align:right; border-right:1px solid #444;">${rowLabel}</td>`;
            
            for (const val of headWeights[i]) {
                const intensity = Math.max(0.0, Math.min(1.0, val));
                const r = 255;
                const g = Math.floor(255 - (180 * intensity));
                const b = Math.floor(200 - (200 * intensity));
                const textCol = intensity < 0.5 ? "black" : "white";
                
                html += `
                    <td style="padding:1px;">
                        <div class="heatmap-cell" style="background:rgb(${r},${g},${b}); color:${textCol}; width:100%; padding:5px; border-radius:4px;">
                            ${val.toFixed(2)}
                        </div>
                    </td>
                `;
            }
            html += `</tr>`;
        }

        html += `</table></div>`;
        document.getElementById("att-visual-results").innerHTML = html;
    }
};
