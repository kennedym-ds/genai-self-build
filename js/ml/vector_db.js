// JS Port of SimpleVectorDB

class TextVectorDB {
    constructor(dimensions = 64) {
        this.dimensions = dimensions;
        this.index = new Map(); // id -> vector
    }

    add(id, vector) {
        if (vector.length !== this.dimensions) {
            console.warn(`Vector dimension mismatch. Expected ${this.dimensions}`);
            return;
        }
        this.index.set(id, vector);
    }

    search(queryVector, topK = 3) {
        const results = [];
        for (const [id, vector] of this.index.entries()) {
            const score = this._cosineSimilarity(queryVector, vector);
            results.push({ id, score });
        }
        
        // Sort descending
        results.sort((a, b) => b.score - a.score);
        return results.slice(0, topK);
    }

    _cosineSimilarity(vecA, vecB) {
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
