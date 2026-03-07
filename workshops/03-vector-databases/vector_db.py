"""
🎯 Workshop 3: Vector Databases (Similarity Search at Scale)
=============================================================

📚 THE LIBRARY ANALOGY:
Our alien friend now has a MAP of word meanings (embeddings from Workshop 2).
But there's a problem: the map has MILLIONS of locations!

When looking for words similar to "king", checking every single word is too slow.
The alien needs a MAGIC LIBRARY where similar books are on nearby shelves.

Imagine a library where:
    - You walk to the "royalty" section → find king, queen, prince nearby
    - You walk to the "animals" section → find cat, dog, pet nearby
    - No need to check EVERY book in the library!

THIS IS EXACTLY WHAT VECTOR DATABASES DO.
They organize vectors so similar ones can be found FAST, without checking all of them.

THREE STRATEGIES (from simple to smart):
1. Flat (Brute Force): Check every vector - accurate but O(n) slow
2. LSH (Locality Sensitive Hashing): Hash similar vectors to same buckets - approximate but fast
3. IVF (Inverted File Index): Cluster vectors, only search nearby clusters - balanced

📍 THIS BUILDS ON WORKSHOPS 1 & 2:
- Workshop 1: Text → Token IDs
- Workshop 2: Token IDs → Vectors with meaning
- Workshop 3: Vectors → Fast similarity search (THIS!)

Usage:
    python vector_db.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time


class SimpleVectorDB:
    """
    A multi-strategy vector database for similarity search.
    
    Stores vectors and enables fast nearest-neighbor search using different
    indexing strategies.
    
    Example usage:
        db = SimpleVectorDB(strategy='ivf', dimensions=50)
        
        # Add vectors with IDs
        db.add("doc1", np.array([0.1, 0.2, ...]))
        db.add("doc2", np.array([0.3, 0.1, ...]))
        
        # Search for similar vectors
        results = db.search(query_vector, top_k=5)
        # [("doc1", 0.95), ("doc2", 0.82), ...]
    """
    
    def __init__(self, strategy: str = 'flat', dimensions: int = 50, **kwargs):
        """
        Initialize the vector database.
        
        Args:
            strategy: One of 'flat', 'lsh', or 'ivf'
            dimensions: Size of the vectors to store
            **kwargs: Strategy-specific parameters
                - lsh: num_tables (default 10), num_bits (default 8)
                - ivf: num_clusters (default 10)
        """
        if strategy not in ['flat', 'lsh', 'ivf']:
            raise ValueError(f"Strategy must be 'flat', 'lsh', or 'ivf', got '{strategy}'")
        
        self.strategy = strategy
        self.dimensions = dimensions
        
        # Core storage: id -> vector
        self.vectors: Dict[str, np.ndarray] = {}
        self.id_list: List[str] = []  # Ordered list for matrix operations
        
        # Strategy-specific parameters and structures
        self.index_built = False
        
        if strategy == 'lsh':
            self.num_tables = kwargs.get('num_tables', 10)
            self.num_bits = kwargs.get('num_bits', 8)
            self._init_lsh()
        elif strategy == 'ivf':
            self.num_clusters = kwargs.get('num_clusters', 10)
            self.centroids: Optional[np.ndarray] = None
            self.cluster_assignments: Dict[int, List[str]] = defaultdict(list)
            self.nprobe = kwargs.get('nprobe', 3)  # How many clusters to search
    
    # =========================================================================
    # PART 0: CORE OPERATIONS (Add, Get, Distance)
    # =========================================================================
    
    def add(self, id: str, vector: np.ndarray) -> None:
        """
        Add a vector to the database.
        
        Args:
            id: Unique identifier for this vector
            vector: numpy array of shape (dimensions,)
        """
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} dimensions, got {len(vector)}")
        
        # Normalize vector (makes cosine similarity = dot product)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        self.vectors[id] = vector.copy()
        if id not in self.id_list:
            self.id_list.append(id)
        
        # Mark index as needing rebuild
        self.index_built = False
    
    def add_batch(self, ids: List[str], vectors: np.ndarray) -> None:
        """
        Add multiple vectors at once.
        
        Args:
            ids: List of unique identifiers
            vectors: numpy array of shape (n, dimensions)
        """
        for id, vector in zip(ids, vectors):
            self.add(id, vector)
    
    def get(self, id: str) -> Optional[np.ndarray]:
        """Get a vector by its ID."""
        return self.vectors.get(id)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Since vectors are normalized, this is just the dot product!
        
        Returns:
            Similarity score in range [-1, 1], higher = more similar
        """
        return float(np.dot(vec1, vec2))
    
    def _get_all_vectors_matrix(self) -> np.ndarray:
        """Get all vectors as a matrix for batch operations."""
        if not self.id_list:
            return np.array([]).reshape(0, self.dimensions)
        return np.array([self.vectors[id] for id in self.id_list])
    
    # =========================================================================
    # PART 1: FLAT (BRUTE FORCE) SEARCH
    # =========================================================================
    
    def _search_flat(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Brute force search: compare query to ALL vectors.
        
        📖 THE NAIVE APPROACH:
        Like checking every book in the library one by one.
        Guaranteed to find the best match, but SLOW for large databases.
        
        Time complexity: O(n * d) where n = num vectors, d = dimensions
        
        Returns:
            List of (id, similarity) tuples, sorted by similarity descending
        """
        if not self.vectors:
            return []
        
        # Normalize query
        query = query / (np.linalg.norm(query) + 1e-10)
        
        # Compute similarity to all vectors at once (matrix multiplication)
        all_vectors = self._get_all_vectors_matrix()
        similarities = all_vectors @ query  # Dot product with all vectors
        
        # Get top-k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency when k << n
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Build results
        results = []
        for idx in top_indices[:top_k]:
            results.append((self.id_list[idx], float(similarities[idx])))
        
        return results
    
    # =========================================================================
    # PART 2: LSH (LOCALITY SENSITIVE HASHING)
    # =========================================================================
    
    def _init_lsh(self) -> None:
        """
        Initialize LSH hash tables and random hyperplanes.
        
        🎲 THE HASHING INSIGHT:
        Imagine the alien has magic coins. When flipped near similar words,
        they land the same way! Words in the same "bucket" (same coin pattern)
        are probably similar.
        
        LSH uses random hyperplanes to create hash codes:
        - Each hyperplane divides space in two (like flipping a coin)
        - Similar vectors land on the same side of most hyperplanes
        - So they get similar hash codes → same bucket!
        """
        # Create random hyperplanes for each hash table
        # Each hyperplane is a random vector; sign of dot product gives 0 or 1
        np.random.seed(42)  # For reproducibility
        
        self.hyperplanes = []
        self.hash_tables: List[Dict[str, List[str]]] = []
        
        for _ in range(self.num_tables):
            # Random hyperplanes for this table
            planes = np.random.randn(self.num_bits, self.dimensions)
            # Normalize each hyperplane
            planes = planes / (np.linalg.norm(planes, axis=1, keepdims=True) + 1e-10)
            self.hyperplanes.append(planes)
            
            # Hash table: hash_code -> list of vector IDs
            self.hash_tables.append(defaultdict(list))
    
    def _lsh_hash(self, vector: np.ndarray, table_idx: int) -> str:
        """
        Compute LSH hash code for a vector.
        
        The hash code is a string of bits, where each bit indicates
        which side of a hyperplane the vector falls on.
        
        Example:
            vector on positive side of planes 0, 2, 3 and negative side of 1
            → hash code "1011"
        """
        planes = self.hyperplanes[table_idx]
        
        # Compute dot product with each hyperplane
        projections = planes @ vector
        
        # Convert to binary: positive = 1, negative = 0
        bits = (projections >= 0).astype(int)
        
        # Convert to string hash code
        return ''.join(map(str, bits))
    
    def _build_lsh_index(self) -> None:
        """
        Build LSH index by hashing all vectors into tables.
        
        After this, each hash table maps hash_codes to lists of vector IDs.
        """
        print(f"📚 Building LSH index ({self.num_tables} tables, {self.num_bits} bits)...")
        
        # Clear existing tables
        for table in self.hash_tables:
            table.clear()
        
        # Hash each vector into each table
        for id in self.id_list:
            vector = self.vectors[id]
            for table_idx in range(self.num_tables):
                hash_code = self._lsh_hash(vector, table_idx)
                self.hash_tables[table_idx][hash_code].append(id)
        
        self.index_built = True
        print(f"✅ LSH index built for {len(self.id_list)} vectors")
    
    def _search_lsh(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        LSH search: find candidates via hashing, then rank them.
        
        🎯 THE SPEEDUP:
        Instead of checking ALL vectors, we only check vectors that
        hash to the same buckets as our query. Similar vectors are
        likely in the same buckets!
        
        Trade-off: Might miss some similar vectors (approximate), but FAST.
        
        Returns:
            List of (id, similarity) tuples, sorted by similarity descending
        """
        if not self.index_built:
            self._build_lsh_index()
        
        # Normalize query
        query = query / (np.linalg.norm(query) + 1e-10)
        
        # Collect candidates from all hash tables
        candidates = set()
        for table_idx in range(self.num_tables):
            hash_code = self._lsh_hash(query, table_idx)
            candidates.update(self.hash_tables[table_idx].get(hash_code, []))
        
        if not candidates:
            # No candidates found, fall back to flat search
            return self._search_flat(query, top_k)
        
        # Compute exact similarity for candidates only
        candidate_scores = []
        for id in candidates:
            similarity = self._cosine_similarity(query, self.vectors[id])
            candidate_scores.append((id, similarity))
        
        # Sort by similarity and return top-k
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return candidate_scores[:top_k]
    
    # =========================================================================
    # PART 3: IVF (INVERTED FILE INDEX)
    # =========================================================================
    
    def _build_ivf_index(self) -> None:
        """
        Build IVF index by clustering vectors.
        
        📊 THE CLUSTERING INSIGHT:
        Imagine the alien organizes the library into sections:
        - "Royalty" section contains king, queen, prince, castle...
        - "Animals" section contains cat, dog, pet, mouse...
        - "Tech" section contains computer, data, neural, learning...
        
        To find words similar to "king", just search the "Royalty" section!
        
        Algorithm:
        1. Run k-means to find cluster centroids
        2. Assign each vector to its nearest centroid
        3. At search time, find nearest centroids, search those clusters
        """
        print(f"📊 Building IVF index ({self.num_clusters} clusters)...")
        
        if len(self.id_list) < self.num_clusters:
            print(f"⚠️ Not enough vectors for {self.num_clusters} clusters, using flat search")
            self.num_clusters = max(1, len(self.id_list) // 2)
        
        # Get all vectors as matrix
        all_vectors = self._get_all_vectors_matrix()
        
        if len(all_vectors) == 0:
            return
        
        # Simple k-means clustering
        self.centroids = self._kmeans(all_vectors, self.num_clusters)
        
        # Assign each vector to nearest centroid
        self.cluster_assignments = defaultdict(list)
        
        for i, id in enumerate(self.id_list):
            vector = self.vectors[id]
            # Find nearest centroid
            distances = np.linalg.norm(self.centroids - vector, axis=1)
            nearest_cluster = int(np.argmin(distances))
            self.cluster_assignments[nearest_cluster].append(id)
        
        self.index_built = True
        
        # Print cluster stats
        sizes = [len(self.cluster_assignments[i]) for i in range(self.num_clusters)]
        print(f"✅ IVF index built: {self.num_clusters} clusters, sizes: {sizes}")
    
    def _kmeans(self, vectors: np.ndarray, k: int, max_iters: int = 20) -> np.ndarray:
        """
        Simple k-means clustering.
        
        Returns:
            Cluster centroids of shape (k, dimensions)
        """
        n = len(vectors)
        
        # Initialize centroids randomly from existing vectors
        np.random.seed(42)
        indices = np.random.choice(n, size=min(k, n), replace=False)
        centroids = vectors[indices].copy()
        
        for iteration in range(max_iters):
            # Assign points to nearest centroid
            distances = np.zeros((n, k))
            for j in range(k):
                distances[:, j] = np.linalg.norm(vectors - centroids[j], axis=1)
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                cluster_points = vectors[assignments == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = cluster_points.mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return centroids
    
    def _search_ivf(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        IVF search: find nearest clusters, then search within them.
        
        🎯 THE SPEEDUP:
        Instead of searching ALL vectors, we:
        1. Find the nprobe nearest cluster centroids
        2. Only search vectors in those clusters
        
        Trade-off: Might miss vectors in other clusters (approximate), but FAST.
        """
        if not self.index_built or self.centroids is None:
            self._build_ivf_index()
        
        if self.centroids is None or len(self.centroids) == 0:
            return self._search_flat(query, top_k)
        
        # Normalize query
        query = query / (np.linalg.norm(query) + 1e-10)
        
        # Find nearest cluster centroids
        centroid_distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(centroid_distances)[:self.nprobe]
        
        # Collect candidates from nearest clusters
        candidates = []
        for cluster_idx in nearest_clusters:
            candidates.extend(self.cluster_assignments[cluster_idx])
        
        if not candidates:
            return []
        
        # Compute exact similarity for candidates
        candidate_scores = []
        for id in candidates:
            similarity = self._cosine_similarity(query, self.vectors[id])
            candidate_scores.append((id, similarity))
        
        # Sort by similarity and return top-k
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return candidate_scores[:top_k]
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def build_index(self) -> None:
        """Build the search index for the current strategy."""
        if self.strategy == 'flat':
            # Flat search doesn't need an index
            self.index_built = True
            print(f"✅ Flat index ready ({len(self.vectors)} vectors)")
        elif self.strategy == 'lsh':
            self._build_lsh_index()
        elif self.strategy == 'ivf':
            self._build_ivf_index()
    
    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for the most similar vectors to the query.
        
        Args:
            query: Query vector of shape (dimensions,)
            top_k: Number of results to return
            
        Returns:
            List of (id, similarity) tuples, sorted by similarity descending
        """
        if len(query) != self.dimensions:
            raise ValueError(f"Query must have {self.dimensions} dimensions, got {len(query)}")
        
        if self.strategy == 'flat':
            return self._search_flat(query, top_k)
        elif self.strategy == 'lsh':
            return self._search_lsh(query, top_k)
        elif self.strategy == 'ivf':
            return self._search_ivf(query, top_k)
        
        return []
    
    def size(self) -> int:
        """Return the number of vectors in the database."""
        return len(self.vectors)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        stats = {
            'strategy': self.strategy,
            'num_vectors': len(self.vectors),
            'dimensions': self.dimensions,
            'index_built': self.index_built,
        }
        
        if self.strategy == 'lsh':
            stats['num_tables'] = self.num_tables
            stats['num_bits'] = self.num_bits
            if self.index_built:
                # Count non-empty buckets
                total_buckets = sum(len(table) for table in self.hash_tables)
                stats['total_buckets'] = total_buckets
        
        elif self.strategy == 'ivf':
            stats['num_clusters'] = self.num_clusters
            stats['nprobe'] = self.nprobe
            if self.index_built:
                sizes = [len(self.cluster_assignments[i]) for i in range(self.num_clusters)]
                stats['cluster_sizes'] = sizes
        
        return stats


# =============================================================================
# DEMO: SEE VECTOR SEARCH IN ACTION!
# =============================================================================

def demo():
    """
    Demonstrate vector database operations and compare strategies.
    """
    print("\n" + "=" * 70)
    print("📚 VECTOR DATABASE DEMO")
    print("=" * 70)
    
    # Create sample document embeddings (simulating output from Workshop 2)
    np.random.seed(42)
    num_docs = 1000
    dimensions = 50
    
    print(f"\n📄 Creating {num_docs} sample document vectors ({dimensions}D)...")
    
    # Create clustered data to simulate real embeddings
    # Cluster 1: "Technology" documents
    tech_vecs = np.random.randn(300, dimensions) * 0.3
    tech_vecs[:, 0] += 2  # Shift in one direction
    
    # Cluster 2: "Science" documents  
    science_vecs = np.random.randn(300, dimensions) * 0.3
    science_vecs[:, 1] += 2  # Shift in another direction
    
    # Cluster 3: "Arts" documents
    arts_vecs = np.random.randn(200, dimensions) * 0.3
    arts_vecs[:, 2] += 2  # Shift in third direction
    
    # Cluster 4: "Sports" documents
    sports_vecs = np.random.randn(200, dimensions) * 0.3
    sports_vecs[:, 3] += 2  # Shift in fourth direction
    
    all_vectors = np.vstack([tech_vecs, science_vecs, arts_vecs, sports_vecs])
    doc_ids = [f"doc_{i}" for i in range(num_docs)]
    doc_categories = (["tech"] * 300 + ["science"] * 300 + 
                      ["arts"] * 200 + ["sports"] * 200)
    
    # Normalize vectors
    all_vectors = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-10)
    
    # =========================================================================
    # Compare strategies
    # =========================================================================
    
    strategies = ['flat', 'lsh', 'ivf']
    query = all_vectors[0]  # Use first tech document as query
    query_category = doc_categories[0]
    
    print(f"\n🔍 Query: {doc_ids[0]} (category: {query_category})")
    print("=" * 70)
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy.upper()} strategy...")
        print("-" * 50)
        
        # Create and populate database
        if strategy == 'lsh':
            db = SimpleVectorDB(strategy=strategy, dimensions=dimensions, 
                               num_tables=10, num_bits=8)
        elif strategy == 'ivf':
            db = SimpleVectorDB(strategy=strategy, dimensions=dimensions,
                               num_clusters=20, nprobe=3)
        else:
            db = SimpleVectorDB(strategy=strategy, dimensions=dimensions)
        
        # Add vectors
        start_time = time.time()
        db.add_batch(doc_ids, all_vectors)
        add_time = time.time() - start_time
        
        # Build index
        start_time = time.time()
        db.build_index()
        build_time = time.time() - start_time
        
        # Search
        start_time = time.time()
        num_searches = 100
        for _ in range(num_searches):
            results = db.search(query, top_k=5)
        search_time = (time.time() - start_time) / num_searches * 1000  # ms per search
        
        # Report results
        print(f"  ⏱️ Add time: {add_time*1000:.2f}ms")
        print(f"  ⏱️ Index build time: {build_time*1000:.2f}ms")
        print(f"  ⏱️ Search time: {search_time:.3f}ms (avg over {num_searches} searches)")
        
        print(f"\n  🎯 Top 5 results:")
        for id, score in results:
            idx = doc_ids.index(id)
            cat = doc_categories[idx]
            match = "✅" if cat == query_category else "❌"
            print(f"      {id} ({cat}): {score:.4f} {match}")
        
        # Count correct category in top 5
        top_categories = [doc_categories[doc_ids.index(id)] for id, _ in results]
        accuracy = sum(1 for c in top_categories if c == query_category) / len(results)
        print(f"\n  📈 Category accuracy (top 5): {accuracy*100:.0f}%")
    
    # =========================================================================
    # Key insights
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70)
    print("""
    📖 FLAT (Brute Force):
       → Checks EVERY vector - always finds the best match
       → O(n) time complexity - too slow for millions of vectors
       → Use when: Small datasets, need perfect accuracy
    
    🎲 LSH (Locality Sensitive Hashing):
       → Hashes similar vectors to same buckets
       → O(1) to O(n) depending on bucket distribution
       → Trade-off: Might miss some similar vectors
       → Use when: Very large datasets, speed > perfect accuracy
    
    📊 IVF (Inverted File Index):
       → Clusters vectors, searches only nearby clusters
       → O(k * cluster_size) where k = nprobe
       → Good balance of speed and accuracy
       → Use when: Large datasets, need good accuracy
    
    🔗 REAL-WORLD USAGE:
       - Pinecone, Weaviate, Qdrant use IVF + other optimizations
       - FAISS (Facebook) supports all these strategies
       - OpenAI's embeddings + vector DB = semantic search!
    
    📍 NEXT UP: Workshop 4 - Attention!
       How does the model know which words to focus on?
    """)


if __name__ == "__main__":
    demo()
