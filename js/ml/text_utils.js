// Shared text + scene utility helpers for UI pages
const TextUtils = {
    hashStr(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) hash = ((hash << 5) - hash) + str.charCodeAt(i);
        return Math.abs(hash);
    },

    makeVector(text, dim = 64) {
        let seed = this.hashStr(text);
        const vec = [];
        let sumSq = 0;
        let x = Math.sin(seed) * 10000;

        for (let d = 0; d < dim; d++) {
            x = Math.sin(x) * 10000;
            const val = ((x - Math.floor(x)) * 2) - 1;
            vec.push(val);
            sumSq += val * val;
        }

        const mag = Math.sqrt(sumSq);
        return vec.map(v => v / mag);
    },

    cosSim(a, b) {
        let dot = 0;
        let na = 0;
        let nb = 0;

        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }

        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    },

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

window.TextUtils = TextUtils;
