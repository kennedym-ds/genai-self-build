"""
🎯 Workshop 2: Words to Meaning (Embeddings)
=============================================

🗺️ THE MAP ANALOGY:
Imagine our alien friend from Workshop 1 has mastered reading symbols (tokenization).
Now they face a new challenge: understanding what words MEAN.

The alien decides to create a MAP of Earth languages. On this map:
    - Words that mean similar things are placed CLOSE together
    - "King" lives near "queen", "prince", "royalty"
    - "Cat" lives near "dog", "pet", "animal"
    - "Happy" lives far from "sad"

But here's the magical part: the alien discovers that meaning follows DIRECTIONS!
    - The direction from "king" to "queen" is the same as "man" to "woman"
    - So: king - man + woman = queen (it's like vector math!)

THIS IS EXACTLY WHAT EMBEDDINGS DO.
Each word becomes a point in space (a vector of numbers), where
the POSITION captures the word's meaning relative to all other words.

THREE STRATEGIES (from simple to smart):
1. Random: Assign random coordinates (baseline - no real meaning!)
2. Co-occurrence: Words appearing together get similar vectors (context = meaning)
3. Prediction-based: Learn vectors by predicting neighboring words (skip-gram style)

📍 THIS BUILDS ON WORKSHOP 1:
After tokenizing text into IDs, we now give those IDs MEANING by placing
them in a semantic space where similar words cluster together.

Usage:
    python embeddings.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import re


class SimpleEmbedding:
    """
    A multi-strategy word embedding model that learns word vectors from text.
    
    Embeddings turn words into vectors where:
        - Similar words have similar vectors
        - Distance represents meaning difference
        - Directions encode relationships (king - man + woman ≈ queen)
    
    Example usage:
        embedder = SimpleEmbedding(strategy='cooccurrence', dimensions=50)
        embedder.train(["The king and queen ruled the kingdom."])
        
        # Find similar words
        embedder.most_similar("king")  # [("queen", 0.89), ("kingdom", 0.72), ...]
        
        # Solve analogies
        embedder.analogy("king", "queen", "man")  # "woman"
    """
    
    def __init__(self, strategy: str = 'cooccurrence', dimensions: int = 50):
        """
        Initialize the embedding model.
        
        Args:
            strategy: One of 'random', 'cooccurrence', or 'prediction'
            dimensions: Size of the embedding vectors (more = more expressive, but slower)
        """
        if strategy not in ['random', 'cooccurrence', 'prediction']:
            raise ValueError(f"Strategy must be 'random', 'cooccurrence', or 'prediction', got '{strategy}'")
        
        self.strategy = strategy
        self.dimensions = dimensions
        
        # Core data structures
        self.vocab: Dict[str, int] = {}           # word -> index
        self.inverse_vocab: Dict[int, str] = {}   # index -> word
        self.embeddings: Optional[np.ndarray] = None  # shape: (vocab_size, dimensions)
        
        # Training statistics
        self.word_counts: Counter = Counter()
        self.is_trained = False
    
    # =========================================================================
    # PART 0: TEXT PREPROCESSING (Same as Workshop 1)
    # =========================================================================
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Split text into lowercase words.
        
        Example:
            text = "The King's crown!"
            output = ["the", "king", "s", "crown"]
        """
        return re.findall(r'\b\w+\b', text.lower())
    
    def _build_vocab(self, corpus: List[str], min_count: int = 1) -> None:
        """
        Build vocabulary from corpus, counting word frequencies.
        
        Example:
            corpus = ["the cat sat", "the dog sat"]
            vocab = {"the": 0, "sat": 1, "cat": 2, "dog": 3}
            word_counts = {"the": 2, "sat": 2, "cat": 1, "dog": 1}
        """
        # Count all words
        for text in corpus:
            words = self._tokenize(text)
            self.word_counts.update(words)
        
        # Build vocabulary (only words appearing min_count times)
        idx = 0
        for word, count in self.word_counts.most_common():
            if count >= min_count:
                self.vocab[word] = idx
                self.inverse_vocab[idx] = word
                idx += 1
        
        print(f"📚 Vocabulary: {len(self.vocab)} words")
    
    # =========================================================================
    # PART 1: RANDOM EMBEDDINGS (Baseline - No Real Meaning!)
    # =========================================================================
    
    def _train_random(self, corpus: List[str]) -> None:
        """
        Assign random vectors to each word (baseline strategy).
        
        🎲 THE RANDOM BASELINE:
        Imagine our alien just throws darts at a map to place words.
        "King" might end up next to "banana" - pure chance!
        
        This is our CONTROL: if random embeddings work for a task,
        the task doesn't really need semantic understanding.
        
        Why include this?
            - Shows that random vectors capture NO meaning
            - Helps appreciate what learned embeddings achieve
            - Useful for debugging and comparison
        """
        # Step 1: Build vocabulary
        self._build_vocab(corpus)
        
        # Step 2: Create random vectors for each word
        # We use a fixed seed for reproducibility
        np.random.seed(42)
        
        vocab_size = len(self.vocab)
        self.embeddings = np.random.randn(vocab_size, self.dimensions)
        
        # Step 3: Normalize vectors to unit length
        # This makes cosine similarity == dot product (faster!)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-10)
        
        self.is_trained = True
        print(f"🎲 Random embeddings: {vocab_size} words × {self.dimensions} dimensions")
    
    # =========================================================================
    # PART 2: CO-OCCURRENCE EMBEDDINGS (Context = Meaning)
    # =========================================================================
    
    def _train_cooccurrence(self, corpus: List[str], window_size: int = 5, **kwargs) -> None:
        """
        Learn embeddings from word co-occurrence patterns.
        
        🔗 THE CO-OCCURRENCE INSIGHT:
        "You shall know a word by the company it keeps" - J.R. Firth (1957)
        
        If two words appear in similar contexts, they probably mean similar things:
            - "The ___ sat on the mat" → cat, dog, baby (similar contexts!)
            - "The ___ is shining" → sun, star, light (similar contexts!)
        
        Algorithm:
            1. Build a word-word co-occurrence matrix
            2. Apply SVD to reduce dimensions (like PCA for text!)
            3. Use the reduced vectors as embeddings
        
        Why SVD?
            - The raw matrix is HUGE (vocab × vocab)
            - SVD finds the most important "directions" of meaning
            - Dimensionality reduction acts as regularization
        """
        # Step 1: Build vocabulary
        self._build_vocab(corpus, min_count=1)
        
        vocab_size = len(self.vocab)
        
        # Step 2: Build co-occurrence matrix
        # cooccur[i][j] = how often word i appears near word j
        print(f"🔗 Building co-occurrence matrix (window={window_size})...")
        cooccur = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        for text in corpus:
            words = self._tokenize(text)
            word_ids = [self.vocab[w] for w in words if w in self.vocab]
            
            # For each word, look at its neighbors within the window
            for i, center_id in enumerate(word_ids):
                # Look at words to the left
                for j in range(max(0, i - window_size), i):
                    context_id = word_ids[j]
                    # Weight by distance (closer words matter more)
                    distance = i - j
                    weight = 1.0 / distance
                    cooccur[center_id][context_id] += weight
                    cooccur[context_id][center_id] += weight  # Symmetric
        
        # Step 3: Apply PPMI (Positive Pointwise Mutual Information)
        # This transforms raw counts into a measure of "meaningful" co-occurrence
        print("📊 Computing PPMI transform...")
        cooccur = self._apply_ppmi(cooccur)
        
        # Step 4: Apply SVD to reduce dimensions
        print(f"🔬 Reducing to {self.dimensions} dimensions via SVD...")
        self.embeddings = self._svd_reduce(cooccur, self.dimensions)
        
        self.is_trained = True
        print(f"✅ Co-occurrence embeddings: {vocab_size} words × {self.dimensions} dimensions")
    
    def _apply_ppmi(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply Positive Pointwise Mutual Information transform.
        
        PMI tells us if two words appear together MORE than chance would predict.
            PMI(x, y) = log( P(x,y) / (P(x) * P(y)) )
        
        PPMI just clips negative values to 0 (we only care about positive associations).
        
        Example:
            - "machine" + "learning" have high PPMI (they co-occur a lot!)
            - "machine" + "banana" have low/zero PPMI (rare together)
        """
        # Add small epsilon to avoid log(0)
        matrix = matrix + 1e-10
        
        # Compute marginal probabilities
        total = matrix.sum()
        row_sums = matrix.sum(axis=1, keepdims=True)
        col_sums = matrix.sum(axis=0, keepdims=True)
        
        # Expected co-occurrence under independence
        expected = (row_sums @ col_sums) / total
        
        # PMI = log(observed / expected)
        pmi = np.log2(matrix / expected + 1e-10)
        
        # PPMI = max(0, PMI)
        ppmi = np.maximum(pmi, 0)
        
        return ppmi
    
    def _svd_reduce(self, matrix: np.ndarray, n_components: int) -> np.ndarray:
        """
        Reduce matrix dimensions using Singular Value Decomposition.
        
        SVD decomposes matrix M = U × S × V^T
        We keep only the top n_components singular values/vectors.
        
        The resulting U × S gives us word embeddings that capture
        the most important patterns of co-occurrence.
        """
        # Use numpy's built-in SVD (for simplicity; scipy is faster for large matrices)
        try:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback for numerical issues
            print("⚠️ SVD failed, using random initialization")
            return np.random.randn(matrix.shape[0], n_components) * 0.1
        
        # Take top n_components
        n_components = min(n_components, len(S))
        U_reduced = U[:, :n_components]
        S_reduced = S[:n_components]
        
        # Weight by sqrt of singular values (empirically works well)
        embeddings = U_reduced * np.sqrt(S_reduced)
        
        # Normalize to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)
        
        return embeddings
    
    # =========================================================================
    # PART 3: PREDICTION-BASED EMBEDDINGS (Learning from Context)
    # =========================================================================
    
    def _train_prediction(self, corpus: List[str], 
                          window_size: int = 2,
                          epochs: int = 50,
                          learning_rate: float = 0.025,
                          negative_samples: int = 10) -> None:
        """
        Learn embeddings by predicting context words (simplified Skip-gram).
        
        🎯 THE PREDICTION INSIGHT:
        Instead of counting co-occurrences, we LEARN vectors that are
        good at predicting which words appear near each other.
        
        Skip-gram idea:
            - Given "king", predict that "queen" and "royal" are nearby
            - But NOT that "banana" or "computer" are nearby
        
        This is a SIMPLIFIED version of Word2Vec's Skip-gram with negative sampling:
            1. For each word, create positive examples (word, context_word)
            2. Create negative examples (word, random_word)
            3. Train to distinguish positive from negative
        
        Why this works:
            - Words in similar contexts get pushed to have similar vectors
            - Negative sampling prevents all vectors from collapsing together
        """
        # Step 1: Build vocabulary
        self._build_vocab(corpus, min_count=1)
        vocab_size = len(self.vocab)
        
        # Step 2: Initialize embeddings randomly
        np.random.seed(42)
        self.embeddings = np.random.randn(vocab_size, self.dimensions) * 0.1
        context_embeddings = np.random.randn(vocab_size, self.dimensions) * 0.1
        
        # Step 3: Create word frequency distribution for negative sampling
        # More frequent words are sampled more often (raised to 0.75 power as in Word2Vec)
        word_freqs = np.array([self.word_counts[self.inverse_vocab[i]] 
                               for i in range(vocab_size)], dtype=np.float64)
        word_freqs = word_freqs ** 0.75
        neg_sample_probs = word_freqs / word_freqs.sum()
        
        # Step 4: Generate training pairs
        print(f"🎯 Generating training pairs (window={window_size})...")
        training_pairs = []
        
        for text in corpus:
            words = self._tokenize(text)
            word_ids = [self.vocab[w] for w in words if w in self.vocab]
            
            for i, center_id in enumerate(word_ids):
                # Context words within window
                start = max(0, i - window_size)
                end = min(len(word_ids), i + window_size + 1)
                
                for j in range(start, end):
                    if j != i:
                        context_id = word_ids[j]
                        training_pairs.append((center_id, context_id))
        
        print(f"📝 Training on {len(training_pairs)} word pairs for {epochs} epochs...")
        
        # Step 5: Train with SGD
        for epoch in range(epochs):
            np.random.shuffle(training_pairs)
            total_loss = 0.0
            
            for center_id, context_id in training_pairs:
                # Positive example: center should predict context
                loss, grad_center, grad_context = self._skip_gram_step(
                    self.embeddings[center_id],
                    context_embeddings[context_id],
                    label=1
                )
                total_loss += loss
                
                # Update embeddings
                self.embeddings[center_id] -= learning_rate * grad_center
                context_embeddings[context_id] -= learning_rate * grad_context
                
                # Negative samples: center should NOT predict random words
                neg_ids = np.random.choice(vocab_size, size=negative_samples, 
                                          p=neg_sample_probs)
                for neg_id in neg_ids:
                    if neg_id != context_id:
                        loss, grad_center, grad_neg = self._skip_gram_step(
                            self.embeddings[center_id],
                            context_embeddings[neg_id],
                            label=0
                        )
                        total_loss += loss
                        self.embeddings[center_id] -= learning_rate * grad_center
                        context_embeddings[neg_id] -= learning_rate * grad_neg
            
            avg_loss = total_loss / len(training_pairs)
            print(f"  Epoch {epoch + 1}/{epochs}: avg_loss = {avg_loss:.4f}")
        
        # Step 6: Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-10)
        
        self.is_trained = True
        print(f"✅ Prediction embeddings: {vocab_size} words × {self.dimensions} dimensions")
    
    def _skip_gram_step(self, center_vec: np.ndarray, context_vec: np.ndarray, 
                        label: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute loss and gradients for one skip-gram training step.
        
        Uses sigmoid cross-entropy loss:
            - label=1: want dot(center, context) to be HIGH (similar)
            - label=0: want dot(center, context) to be LOW (different)
        
        Returns:
            loss: scalar loss value
            grad_center: gradient for center embedding
            grad_context: gradient for context embedding
        """
        # Compute dot product and sigmoid
        score = np.dot(center_vec, context_vec)
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(score, -20, 20)))
        
        # Loss: -log(sigmoid) for positive, -log(1-sigmoid) for negative
        if label == 1:
            loss = -np.log(sigmoid + 1e-10)
        else:
            loss = -np.log(1 - sigmoid + 1e-10)
        
        # Gradient: (sigmoid - label) * other_vector
        grad_factor = sigmoid - label
        grad_center = grad_factor * context_vec
        grad_context = grad_factor * center_vec
        
        return loss, grad_center, grad_context
    
    # =========================================================================
    # PUBLIC METHODS: Training and Querying
    # =========================================================================
    
    def train(self, corpus: List[str], **kwargs) -> None:
        """
        Train embeddings on a corpus of texts.
        
        Args:
            corpus: List of text documents
            **kwargs: Strategy-specific parameters (window_size, epochs, etc.)
        """
        print(f"\n{'='*60}")
        print(f"🗺️ Training {self.strategy.upper()} embeddings...")
        print(f"{'='*60}")
        
        if self.strategy == 'random':
            self._train_random(corpus)
        elif self.strategy == 'cooccurrence':
            self._train_cooccurrence(corpus, **kwargs)
        elif self.strategy == 'prediction':
            self._train_prediction(corpus, **kwargs)
    
    def get_vector(self, word: str) -> np.ndarray:
        """
        Get the embedding vector for a word.
        
        Args:
            word: The word to look up
            
        Returns:
            numpy array of shape (dimensions,)
            
        Raises:
            ValueError: If word not in vocabulary
            
        Example:
            vec = embedder.get_vector("king")
            print(vec.shape)  # (50,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call train() first.")
        
        word = word.lower()
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary. "
                           f"Vocab size: {len(self.vocab)}")
        
        idx = self.vocab[word]
        return self.embeddings[idx].copy()
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between two words.
        
        Cosine similarity = cos(angle between vectors) = dot(a,b) / (||a|| * ||b||)
        
        Range: -1 (opposite) to 1 (identical)
            - 1.0: identical meaning
            - 0.0: unrelated
            - -1.0: opposite meaning (rare in practice)
        
        Example:
            embedder.similarity("king", "queen")  # ~0.85 (similar!)
            embedder.similarity("king", "banana")  # ~0.1 (unrelated)
        """
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        # Since vectors are normalized, cosine sim = dot product
        return float(np.dot(vec1, vec2))
    
    def most_similar(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar words to a given word.
        
        Args:
            word: The query word
            top_n: Number of results to return
            
        Returns:
            List of (word, similarity) tuples, sorted by similarity
            
        Example:
            embedder.most_similar("king", top_n=3)
            # [("queen", 0.89), ("prince", 0.82), ("royal", 0.78)]
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call train() first.")
        
        query_vec = self.get_vector(word)
        
        # Compute similarity to all words (matrix-vector product)
        similarities = self.embeddings @ query_vec
        
        # Get top_n + 1 (to exclude the query word itself)
        top_indices = np.argsort(similarities)[::-1][:top_n + 1]
        
        results = []
        for idx in top_indices:
            similar_word = self.inverse_vocab[idx]
            if similar_word != word.lower():
                results.append((similar_word, float(similarities[idx])))
                if len(results) >= top_n:
                    break
        
        return results
    
    def analogy(self, a: str, b: str, c: str, top_n: int = 1) -> List[Tuple[str, float]]:
        """
        Solve word analogies: a is to b as c is to ?
        
        Uses vector arithmetic: result = b - a + c
        The word closest to this result vector is the answer.
        
        📐 THE ANALOGY MAGIC:
        If embeddings capture meaning, then:
            - "king" - "man" + "woman" ≈ "queen"
            - "paris" - "france" + "germany" ≈ "berlin"
            - "walking" - "walk" + "run" ≈ "running"
        
        Args:
            a, b, c: Words for the analogy "a is to b as c is to ?"
            top_n: Number of candidate answers to return
            
        Returns:
            List of (word, similarity) tuples
            
        Example:
            embedder.analogy("king", "queen", "man")  # [("woman", 0.82)]
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call train() first.")
        
        # Get vectors for a, b, c
        vec_a = self.get_vector(a)
        vec_b = self.get_vector(b)
        vec_c = self.get_vector(c)
        
        # Compute target vector: b - a + c
        target = vec_b - vec_a + vec_c
        
        # Normalize target
        target = target / (np.linalg.norm(target) + 1e-10)
        
        # Find closest words (excluding a, b, c)
        similarities = self.embeddings @ target
        top_indices = np.argsort(similarities)[::-1]
        
        exclude = {a.lower(), b.lower(), c.lower()}
        results = []
        
        for idx in top_indices:
            word = self.inverse_vocab[idx]
            if word not in exclude:
                results.append((word, float(similarities[idx])))
                if len(results) >= top_n:
                    break
        
        return results
    
    def get_all_words(self) -> List[str]:
        """Get list of all words in vocabulary."""
        return list(self.vocab.keys())
    
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)


# =============================================================================
# DEMO: SEE EMBEDDINGS IN ACTION!
# =============================================================================

def demo():
    """
    Demonstrate embedding training and querying.
    
    Shows the difference between random, co-occurrence, and prediction-based embeddings.
    """
    
    # Sample corpus about royalty, animals, and technology
    # More repetition = better embeddings (in real life, use millions of sentences!)
    # We repeat key patterns multiple times to strengthen the signal
    corpus = [
        # Royalty domain - repeated patterns to strengthen king/queen association
        "The king and queen ruled the kingdom wisely.",
        "The king and queen lived in the royal castle.",
        "The king and queen attended the royal ceremony.",
        "The prince and princess are the children of the king and queen.",
        "The king wore a golden crown on his head.",
        "The queen wore a silver crown on her head.",
        "The king sat on the royal throne.",
        "The queen sat beside the king on her throne.",
        "The royal king ruled the kingdom.",
        "The royal queen ruled beside the king.",
        "The kingdom loved their king and queen.",
        "The prince will become king and the princess may become queen.",
        
        # Animals domain - repeated patterns for cat/dog association  
        "The cat and dog are popular pets.",
        "The cat and dog played in the yard.",
        "The cat and dog are furry animals.",
        "My pet cat sleeps on the soft bed.",
        "My pet dog runs in the green park.",
        "The cat chased the mouse while the dog barked.",
        "A cat is a pet and a dog is a pet.",
        "The puppy is a young dog and the kitten is a young cat.",
        "Dogs and cats are common household pets.",
        "The fluffy cat and the friendly dog.",
        
        # Technology domain
        "Machine learning transforms data into insights.",
        "Neural networks learn patterns from data.",
        "Deep learning uses neural networks.",
        "The model learns from training data.",
        "Artificial intelligence powers many applications.",
        "Data science combines statistics and programming.",
        "Machine learning and deep learning use data.",
        "Neural networks are used in machine learning.",
        
        # Gender pairs for analogies - key pattern!
        "The man and woman walked together.",
        "The man and woman talked together.",
        "The boy and girl played in the park.",
        "The boy and girl are children.",
        "He is a man and she is a woman.",
        "A king is a man and a queen is a woman.",
        "The king is a royal man.",
        "The queen is a royal woman.",
        "Man and woman are adults.",
        "The man worked while the woman worked.",
        
        # More reinforcement
        "The king rules and the queen rules.",
        "A cat purrs and a dog barks.",
        "The man is strong and the woman is strong.",
    ]
    
    print("\n" + "=" * 70)
    print("🗺️ WORD EMBEDDINGS DEMO")
    print("=" * 70)
    print("\n📚 Training on a small corpus about royalty, animals, and technology.\n")
    
    # =========================================================================
    # Compare the three strategies
    # =========================================================================
    
    strategies = ['random', 'cooccurrence', 'prediction']
    embedders = {}
    
    for strategy in strategies:
        embedder = SimpleEmbedding(strategy=strategy, dimensions=30)
        embedder.train(corpus, window_size=3, epochs=10)
        embedders[strategy] = embedder
        print()
    
    # =========================================================================
    # Test 1: Word Similarity
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("📊 TEST 1: Word Similarity")
    print("=" * 70)
    print("\nComparing similarity scores across strategies:")
    print("(Higher = more similar, max = 1.0)\n")
    
    word_pairs = [
        ("king", "queen"),
        ("cat", "dog"),
        ("king", "cat"),  # Should be low - different domains
        ("learning", "data"),
    ]
    
    print(f"{'Word Pair':<20} {'Random':<12} {'Co-occur':<12} {'Predict':<12}")
    print("-" * 56)
    
    for w1, w2 in word_pairs:
        scores = []
        for strategy in strategies:
            try:
                sim = embedders[strategy].similarity(w1, w2)
                scores.append(f"{sim:.3f}")
            except ValueError:
                scores.append("N/A")
        print(f"{w1+' / '+w2:<20} {scores[0]:<12} {scores[1]:<12} {scores[2]:<12}")
    
    # =========================================================================
    # Test 2: Most Similar Words
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("🔍 TEST 2: Most Similar Words")
    print("=" * 70)
    
    test_words = ["king", "cat", "learning"]
    
    for word in test_words:
        print(f"\n🎯 Words most similar to '{word}':")
        print("-" * 50)
        
        for strategy in strategies:
            try:
                similar = embedders[strategy].most_similar(word, top_n=3)
                similar_str = ", ".join([f"{w} ({s:.2f})" for w, s in similar])
                print(f"  {strategy.capitalize():<15}: {similar_str}")
            except ValueError as e:
                print(f"  {strategy.capitalize():<15}: {e}")
    
    # =========================================================================
    # Test 3: Analogies (The Magic!)
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("✨ TEST 3: Word Analogies")
    print("=" * 70)
    print("\nSolving: 'a is to b as c is to ?'")
    print("(This is where learned embeddings really shine!)\n")
    
    analogies = [
        ("king", "queen", "man"),    # king:queen :: man:? (woman)
        ("cat", "kitten", "dog"),    # cat:kitten :: dog:? (puppy)
    ]
    
    for a, b, c in analogies:
        print(f"❓ {a} : {b} :: {c} : ?")
        for strategy in strategies:
            try:
                results = embedders[strategy].analogy(a, b, c, top_n=3)
                answers = ", ".join([f"{w} ({s:.2f})" for w, s in results])
                print(f"   {strategy.capitalize():<15}: {answers}")
            except ValueError as e:
                print(f"   {strategy.capitalize():<15}: {e}")
        print()
    
    # =========================================================================
    # Insight: Why Learned > Random
    # =========================================================================
    
    print("=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70)
    print("""
    🎲 RANDOM embeddings: Words are scattered randomly in space.
       → Similarity scores are essentially meaningless
       → Analogies give random results
       → Useful as a BASELINE to show that meaning requires learning!
    
    🔗 CO-OCCURRENCE embeddings: "You know a word by its neighbors"
       → Words appearing in similar contexts get similar vectors
       → Captures topical similarity well
       → Fast to compute with SVD
    
    🎯 PREDICTION embeddings: Learn to predict context words
       → Similar to Word2Vec's Skip-gram
       → Often captures more nuanced relationships
       → Works better with more training data
    
    📍 NEXT UP: Workshop 3 - Vector Databases!
       Now that words are vectors, how do we search millions of them fast?
    """)


if __name__ == "__main__":
    demo()
