/**
 * JS Port of SimpleTokenizer (tokenizer.py)
 * Supports 'char', 'word', and basic 'bpe' strategies.
 */
class TextTokenizer {
    constructor(strategy = 'word') {
        this.strategy = strategy;
        this.vocab = new Map();
        this.inverseVocab = new Map();
        this.merges = new Map();
        this.unkTokenId = 0;
        this.vocab.set('<UNK>', this.unkTokenId);
        this.inverseVocab.set(this.unkTokenId, '<UNK>');
    }

    train(corpus, vocabSize) {
        if (this.strategy === 'char') {
            const chars = new Set();
            for (const text of corpus) {
                for (const char of text) {
                    chars.add(char);
                }
            }
            this._buildVocab(Array.from(chars));
        } 
        else if (this.strategy === 'word') {
            const words = new Set();
            for (const text of corpus) {
                const textWords = text.match(/\b\w+\b|[^\w\s]/g) || [];
                for (const word of textWords) {
                    words.add(word.toLowerCase());
                }
            }
            this._buildVocab(Array.from(words));
        } 
        else if (this.strategy === 'bpe') {
            // Simplified Toy BPE implementation for educational purposes
            // Start with char vocab
            const chars = new Set();
            let splits = [];
            
            for (const text of corpus) {
                const words = text.match(/\b\w+\b|[^\w\s]/g) || [];
                for (let word of words) {
                    word = word.toLowerCase();
                    const charsInWord = word.split('');
                    for (const c of charsInWord) chars.add(c);
                    splits.push(charsInWord);
                }
            }
            
            this._buildVocab(Array.from(chars));
            
            // Perform dummy merges up to vocabSize (simplified for browser performance)
            let currentVocabSize = this.vocab.size;
            
            while (currentVocabSize < vocabSize) {
                const pairs = new Map();
                for (const split of splits) {
                    for (let i = 0; i < split.length - 1; i++) {
                        const pair = `${split[i]} ${split[i+1]}`;
                        pairs.set(pair, (pairs.get(pair) || 0) + 1);
                    }
                }
                
                if (pairs.size === 0) break;
                
                // Find most frequent pair
                let bestPair = null;
                let maxCount = -1;
                for (const [pair, count] of pairs.entries()) {
                    if (count > maxCount) {
                        maxCount = count;
                        bestPair = pair;
                    }
                }
                
                if (!bestPair) break;
                
                const [p1, p2] = bestPair.split(' ');
                const newToken = p1 + p2;
                
                this.merges.set(bestPair, newToken);
                
                const newId = currentVocabSize;
                this.vocab.set(newToken, newId);
                this.inverseVocab.set(newId, newToken);
                currentVocabSize++;
                
                // Apply merge to splits
                for (let i = 0; i < splits.length; i++) {
                    const newSplit = [];
                    let j = 0;
                    while (j < splits[i].length) {
                        if (j < splits[i].length - 1 && 
                            splits[i][j] === p1 && 
                            splits[i][j+1] === p2) {
                            newSplit.push(newToken);
                            j += 2;
                        } else {
                            newSplit.push(splits[i][j]);
                            j += 1;
                        }
                    }
                    splits[i] = newSplit;
                }
            }
        }
    }

    _buildVocab(tokens) {
        // Start from 1 because 0 is <UNK>
        let idx = 1;
        for (const token of tokens) {
            this.vocab.set(token, idx);
            this.inverseVocab.set(idx, token);
            idx++;
        }
    }

    encode(text) {
        if (!text) return [];
        
        let tokens = [];
        if (this.strategy === 'char') {
            for (const char of text) {
                tokens.push(this.vocab.get(char) ?? this.unkTokenId);
            }
        } 
        else if (this.strategy === 'word') {
            const words = text.match(/\b\w+\b|[^\w\s]/g) || [];
            for (const word of words) {
                tokens.push(this.vocab.get(word.toLowerCase()) ?? this.unkTokenId);
            }
        } 
        else if (this.strategy === 'bpe') {
            const words = text.match(/\b\w+\b|[^\w\s]/g) || [];
            for (const word of words) {
                let split = word.toLowerCase().split('');
                
                // Keep applying known merges
                let changed = true;
                while (changed) {
                    changed = false;
                    for (let i = 0; i < split.length - 1; i++) {
                        const pair = `${split[i]} ${split[i+1]}`;
                        if (this.merges.has(pair)) {
                            const newToken = this.merges.get(pair);
                            split.splice(i, 2, newToken);
                            changed = true;
                            // Only do one merge per pass to respect merge order
                            break; 
                        }
                    }
                }
                
                for (const t of split) {
                    tokens.push(this.vocab.get(t) ?? this.unkTokenId);
                }
            }
        }
        return tokens;
    }

    decode(tokenIds) {
        const tokens = tokenIds.map(id => this.inverseVocab.get(id) ?? '<UNK>');
        if (this.strategy === 'word') {
            return tokens.join(' ');
        } else {
            return tokens.join('');
        }
    }

    vocabSize() {
        return this.vocab.size;
    }
}
