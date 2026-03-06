// JS Port of SimpleEmbedding (embeddings.py demo logic)
// In the Streamlit app, this was mostly a random hash demonstration to teach cosine similarity.

class TextEmbedding {
    constructor(dimensions = 32) {
        this.dimensions = dimensions;
    }

    // A simple, deterministic string hash function to generate stable "random" vectors
    _hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash);
    }

    // Generate a pseudo-random normalized vector based on the word
    getEmbedding(word) {
        const seed = this._hashString(word.toLowerCase());
        const vector = [];
        
        // Simple PRNG
        let x = Math.sin(seed) * 10000;
        
        let sumSq = 0;
        for (let i = 0; i < this.dimensions; i++) {
            x = Math.sin(x) * 10000;
            const val = x - Math.floor(x);
            // shift to -1 to 1
            const normalizedVal = (val * 2) - 1;
            vector.push(normalizedVal);
            sumSq += normalizedVal * normalizedVal;
        }

        // Normalize (L2)
        const magnitude = Math.sqrt(sumSq);
        return vector.map(v => v / magnitude);
    }

    cosineSimilarity(vecA, vecB) {
        if (vecA.length !== vecB.length) return 0;
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        
        if (normA === 0 || normB === 0) return 0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
