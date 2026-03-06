// JS Port of SimpleAttention (QKV Multi-Head Attention Toy Model)

class TextAttention {
    constructor(embedDim = 64, numHeads = 1) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = Math.floor(embedDim / numHeads);
        
        // In a real network these are learned weight matrices.
        // We simulate this perfectly deterministically for the visualizer.
        this.W_q = this._createMatrix(embedDim, embedDim, 1);
        this.W_k = this._createMatrix(embedDim, embedDim, 2);
        this.W_v = this._createMatrix(embedDim, embedDim, 3);
        this.W_o = this._createMatrix(embedDim, embedDim, 4);
    }

    _createMatrix(rows, cols, seedOffset) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                // Pseudo-random deterministic init
                let val = Math.sin((i * cols + j + seedOffset) * 1000) * 10000;
                row.push((val - Math.floor(val)) * 0.2 - 0.1); 
            }
            matrix.push(row);
        }
        return matrix;
    }

    _matMul(A, B) {
        const result = Array(A.length).fill(0).map(() => Array(B[0].length).fill(0));
        for (let i = 0; i < A.length; i++) {
            for (let j = 0; j < B[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < A[0].length; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
    
    _transpose(A) {
        return A[0].map((_, colIndex) => A.map(row => row[colIndex]));
    }

    _softmax(row) {
        const max = Math.max(...row);
        const exps = row.map(v => Math.exp(v - max)); // Stability
        const sumExp = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sumExp);
    }

    // Input x is an array of shape [seq_len, embedDim]
    forward(x) {
        // Linear projections
        const Q = this._matMul(x, this.W_q);
        const K = this._matMul(x, this.W_k);
        const V = this._matMul(x, this.W_v);
        
        // For numHeads=1 demo
        const K_T = this._transpose(K);
        
        // Q * K^T
        let scores = this._matMul(Q, K_T);
        
        // Scale
        const scale = Math.sqrt(this.headDim);
        scores = scores.map(row => row.map(val => val / scale));
        
        // Softmax
        const weights = scores.map(row => this._softmax(row));
        
        // Multiply by V
        const out = this._matMul(weights, V);
        
        // Final projection
        const finalOut = this._matMul(out, this.W_o);
        
        return {
            output: finalOut,
            weights: weights // Return attention matrix for visualization
        };
    }
}
