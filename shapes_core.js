// 🛸 Shapes Core JS — Full API for Vanilla SPA
// Complete port of shapes_core.py

// --- Tokenizer ---
class ShapeTokenizer {
    constructor(strategy = 'part') {
        this.strategy = strategy;
        this.vocab = {};
        this.inverse_vocab = {};
        this.is_trained = false;
    }

    train(scenes) {
        if (this.strategy === 'pixel') this._train_pixel();
        else if (this.strategy === 'shape') this._train_shape(scenes);
        else this._train_part();
        this.is_trained = true;
    }

    _train_pixel() {
        const colors = ["empty", ...Object.keys(LAYER_COLORS)];
        colors.forEach((c, i) => {
            this.vocab[c] = i;
            this.inverse_vocab[i] = c;
        });
    }

    _train_shape(scenes) {
        this.vocab = { "<UNK>": 0 };
        let idx = 1;
        scenes.forEach(scene => {
            (scene.shapes || []).forEach(s => {
                const key = `${s.type}_${s.color}`;
                if (!(key in this.vocab)) {
                    this.vocab[key] = idx++;
                }
            });
        });
        Object.entries(this.vocab).forEach(([k, v]) => this.inverse_vocab[v] = k);
    }

    _train_part() {
        SHAPE_PARTS.forEach((p, i) => {
            this.vocab[p] = i;
            this.inverse_vocab[i] = p;
        });
    }

    encode(scene) {
        if (this.strategy === 'pixel') return this._encode_pixel(scene);
        if (this.strategy === 'shape') return this._encode_shape(scene);
        return this._encode_part(scene);
    }

    _encode_pixel(scene, gridSize = 10) {
        const grid = Array(gridSize).fill(0).map(() => Array(gridSize).fill(this.vocab["empty"]));
        (scene.shapes || []).forEach(s => {
            const color_id = this.vocab[s.color] !== undefined ? this.vocab[s.color] : this.vocab["empty"];
            const x0 = Math.floor(s.x), y0 = Math.floor(s.y);
            const x1 = Math.min(x0 + Math.ceil(s.w), gridSize);
            const y1 = Math.min(y0 + Math.ceil(s.h), gridSize);
            for (let r = Math.max(0, y0); r < y1; r++) {
                for (let c = Math.max(0, x0); c < x1; c++) {
                    grid[r][c] = color_id;
                }
            }
        });
        return grid.flat();
    }

    _encode_shape(scene) {
        const unk = this.vocab["<UNK>"] || 0;
        return (scene.shapes || []).map(s => {
            const key = `${s.type}_${s.color}`;
            return this.vocab[key] !== undefined ? this.vocab[key] : unk;
        });
    }

    _encode_part(scene) {
        const tokens = [];
        (scene.shapes || []).forEach(s => {
            const parts = SHAPE_DECOMPOSITION[s.type] || ["straight_edge"];
            parts.forEach(p => tokens.push(this.vocab[p] || 0));
        });
        return tokens;
    }

    decode(token_ids) {
        return token_ids.map(id => this.inverse_vocab[id] || "?");
    }

    get vocab_size() {
        return Object.keys(this.vocab).length;
    }
}

// --- Embeddings ---
class ShapeEmbedding {
    constructor(strategy = 'spatial', dimensions = 16) {
        this.strategy = strategy;
        this.dimensions = dimensions;
        this.scene_names = [];
        this.embeddings = [];
    }

    train(scenesDict) {
        this.scene_names = Object.keys(scenesDict);
        this.embeddings = this.scene_names.map(name => {
            return this._compute_embedding(scenesDict[name]);
        });
    }

    _compute_embedding(scene) {
        const shapes = scene.shapes || [];
        const dim = this.dimensions;
        const vec = new Array(dim).fill(0);

        if (shapes.length === 0) return vec;

        // Feature extraction: spread shape properties across dimensions
        shapes.forEach((s, idx) => {
            vec[idx % dim] += s.x / 10.0;
            vec[(idx + 1) % dim] += s.y / 10.0;
            vec[(idx + 2) % dim] += s.w / 5.0;
            vec[(idx + 3) % dim] += s.h / 5.0;
            // Color hash
            let colorHash = 0;
            for (let i = 0; i < s.color.length; i++) colorHash += s.color.charCodeAt(i);
            vec[(idx + 4) % dim] += (colorHash % 10) / 10.0;
            // Type hash
            let typeHash = 0;
            for (let i = 0; i < s.type.length; i++) typeHash += s.type.charCodeAt(i);
            vec[(idx + 5) % dim] += (typeHash % 10) / 10.0;
        });

        // Normalize
        let mag = 0;
        vec.forEach(v => mag += v * v);
        mag = Math.sqrt(mag);
        if (mag > 0) {
            for (let i = 0; i < dim; i++) vec[i] /= mag;
        }

        return vec;
    }

    get_embedding(scene) {
        // If scene is a string (name), look it up
        if (typeof scene === 'string') {
            const idx = this.scene_names.indexOf(scene);
            if (idx >= 0) return this.embeddings[idx];
            // Fallback: compute from SAMPLE_SCENES if available
            if (typeof SAMPLE_SCENES !== 'undefined' && SAMPLE_SCENES[scene]) {
                return this._compute_embedding(SAMPLE_SCENES[scene]);
            }
            return new Array(this.dimensions).fill(0);
        }
        // Otherwise it's a scene dict — compute directly
        return this._compute_embedding(scene);
    }
}

// --- Vector Database ---
class ShapeVectorDB {
    constructor(dimensions = 16) {
        this.dimensions = dimensions;
        this.ids = [];
        this.vectors = [];
    }

    add_batch(ids, vectors) {
        for (let i = 0; i < ids.length; i++) {
            this.ids.push(ids[i]);
            this.vectors.push(vectors[i]);
        }
    }

    search(queryVector, k = 3) {
        const results = [];
        for (let i = 0; i < this.ids.length; i++) {
            const sim = this._cosineSimilarity(queryVector, this.vectors[i]);
            results.push([this.ids[i], sim]);
        }
        results.sort((a, b) => b[1] - a[1]);
        return results.slice(0, k);
    }

    _cosineSimilarity(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na === 0 || nb === 0) return 0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}

// --- Attention ---
class ShapeAttention {
    compute_attention(scene) {
        const shapes = scene.shapes || [];
        const n = shapes.length;
        const labels = shapes.map(s => s.label);
        
        if (n === 0) return [[], []];

        const proximity = this._proximity_scores(shapes);
        const color = this._color_scores(shapes);
        const alignment = this._alignment_scores(shapes);
        const containment = this._containment_scores(shapes);

        const weights = [proximity, color, alignment, containment].map(matrix => 
            matrix.map(row => this._softmax(row))
        );

        // Return as array tuple [weights, labels] to match app.js expectations
        return [weights, labels];
    }

    _softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    _center(s) {
        return [s.x + s.w / 2, s.y + s.h / 2];
    }

    _proximity_scores(shapes) {
        const n = shapes.length;
        const scores = Array(n).fill().map(() => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            const [cx_i, cy_i] = this._center(shapes[i]);
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const [cx_j, cy_j] = this._center(shapes[j]);
                const dist = Math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2);
                scores[i][j] = Math.max(0, 10 - dist);
            }
        }
        return scores;
    }

    _color_scores(shapes) {
        const n = shapes.length;
        const scores = Array(n).fill().map(() => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                scores[i][j] = shapes[i].color === shapes[j].color ? 5.0 : 0.5;
            }
        }
        return scores;
    }

    _alignment_scores(shapes) {
        const n = shapes.length;
        const scores = Array(n).fill().map(() => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            const [cx_i, cy_i] = this._center(shapes[i]);
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const [cx_j, cy_j] = this._center(shapes[j]);
                const x_align = Math.max(0, 2 - Math.abs(cx_i - cx_j));
                const y_align = Math.max(0, 2 - Math.abs(cy_i - cy_j));
                scores[i][j] = x_align + y_align;
            }
        }
        return scores;
    }

    _containment_scores(shapes) {
        const n = shapes.length;
        const scores = Array(n).fill().map(() => Array(n).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const [cx_j, cy_j] = this._center(shapes[j]);
                if (cx_j >= shapes[i].x && cx_j <= shapes[i].x + shapes[i].w &&
                    cy_j >= shapes[i].y && cy_j <= shapes[i].y + shapes[i].h) {
                    scores[i][j] = 5.0;
                } else {
                    scores[i][j] = 0.2;
                }
            }
        }
        return scores;
    }
}
