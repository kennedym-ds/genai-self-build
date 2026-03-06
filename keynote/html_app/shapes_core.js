// 🛸 Shapes Core JS
// Ported from shapes_core.py

(function() {
const { SHAPE_TYPES, LAYER_COLORS, SHAPE_PARTS, SHAPE_DECOMPOSITION } = window.ShapeData;

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
            scene.shapes.forEach(s => {
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
        scene.shapes.forEach(s => {
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
        return scene.shapes.map(s => {
            const key = `${s.type}_${s.color}`;
            return this.vocab[key] !== undefined ? this.vocab[key] : unk;
        });
    }

    _encode_part(scene) {
        const tokens = [];
        scene.shapes.forEach(s => {
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

// --- Embeddings (Fake/Deterministic for Demo) ---
// Since we don't have numpy/SVD in vanilla JS easily, we hardcode 2D positions for the plot based on similarities 
// or implement a very naive PCA. For this specific demo frontend, we just need a deterministic 2D map.
class ShapeEmbedding {
    // simplified for JS demo
    constructor() {
        this.scenePositions = {}; // pre-calculated 2D positions for the map
        this.sceneNames = [];
    }

    train(scenesDict) {
        this.sceneNames = Object.keys(scenesDict);
        // Fake a 2D projection spread in a circle for visual demonstration
        this.sceneNames.forEach((name, i) => {
            const angle = (i / this.sceneNames.length) * Math.PI * 2;
            const r = 0.5 + Math.random() * 0.5;
            this.scenePositions[name] = [Math.cos(angle) * r, Math.sin(angle) * r];
        });
    }

    get_2d_projection() {
        return this.sceneNames.map(n => this.scenePositions[n]);
    }
}

// --- Attention Heatmaps ---
class ShapeAttention {
    compute_attention(scene) {
        const shapes = scene.shapes;
        const n = shapes.length;
        const labels = shapes.map(s => s.label);
        
        if (n === 0) return { weights: [], labels: [] };

        const proximity = this._proximity_scores(shapes);
        const color = this._color_scores(shapes);
        const alignment = this._alignment_scores(shapes);
        const containment = this._containment_scores(shapes);

        // softmax each row
        const weights = [proximity, color, alignment, containment].map(matrix => 
            matrix.map(row => this._softmax(row))
        );

        return { weights, labels };
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
        for (let i=0; i<n; i++) {
            const [cx_i, cy_i] = this._center(shapes[i]);
            for (let j=0; j<n; j++) {
                if (i===j) continue;
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
        for (let i=0; i<n; i++) {
            for (let j=0; j<n; j++) {
                if (i===j) continue;
                scores[i][j] = shapes[i].color === shapes[j].color ? 5.0 : 0.5;
            }
        }
        return scores;
    }

    _alignment_scores(shapes) {
        const n = shapes.length;
        const scores = Array(n).fill().map(() => Array(n).fill(0));
        for (let i=0; i<n; i++) {
            const [cx_i, cy_i] = this._center(shapes[i]);
            for (let j=0; j<n; j++) {
                if (i===j) continue;
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
        for (let i=0; i<n; i++) {
            for (let j=0; j<n; j++) {
                if (i===j) continue;
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

window.ShapesCore = { ShapeTokenizer, ShapeEmbedding, ShapeAttention };
})();
