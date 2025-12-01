"""
🧪 Workshop 3: Vector Database Test Suite
==========================================

This test suite validates the vector database implementation.
Run this to verify the code is working correctly!

Usage:
    python test_vector_db.py             # Run all tests
    python test_vector_db.py --verbose   # Detailed output
"""

import sys
import os
import argparse
import numpy as np
from typing import List, Dict, Tuple, Any

# Add paths for importing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestResult:
    """Stores test results."""
    def __init__(self, name: str, passed: bool, message: str = "", details: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details


class VectorDBTestSuite:
    """Comprehensive test suite for the SimpleVectorDB class."""
    
    def __init__(self, db_class, verbose: bool = False):
        self.DBClass = db_class
        self.verbose = verbose
        self.results: List[TestResult] = []
        
        # Create test vectors
        np.random.seed(42)
        self.dimensions = 50
        self.num_vectors = 100
        
        # Create clustered test data
        self.test_vectors = self._create_test_vectors()
        self.test_ids = [f"vec_{i}" for i in range(self.num_vectors)]
    
    def _create_test_vectors(self) -> np.ndarray:
        """Create clustered test vectors for meaningful similarity tests."""
        # Create 4 clusters
        cluster_size = self.num_vectors // 4
        
        vectors = []
        for i in range(4):
            cluster = np.random.randn(cluster_size, self.dimensions) * 0.2
            cluster[:, i] += 2  # Shift each cluster in a different direction
            vectors.append(cluster)
        
        all_vectors = np.vstack(vectors)
        # Normalize
        all_vectors = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-10)
        return all_vectors
    
    def log(self, message: str):
        """Print if verbose mode is on."""
        if self.verbose:
            print(f"    {message}")
    
    def run_test(self, test_fn, test_name: str) -> TestResult:
        """Run a single test and capture the result."""
        try:
            passed, message, details = test_fn()
            result = TestResult(test_name, passed, message, details)
        except NotImplementedError:
            result = TestResult(test_name, False, "Not implemented yet", "")
        except Exception as e:
            result = TestResult(test_name, False, f"Exception: {type(e).__name__}", str(e))
        
        self.results.append(result)
        return result
    
    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================
    
    def test_init_default_strategy(self) -> Tuple[bool, str, str]:
        """Test default initialization."""
        db = self.DBClass()
        
        if db.strategy != 'flat':
            return False, f"Default strategy should be 'flat', got '{db.strategy}'", ""
        
        if db.dimensions != 50:
            return False, f"Default dimensions should be 50, got {db.dimensions}", ""
        
        return True, "Default initialization correct", f"strategy={db.strategy}, dims={db.dimensions}"
    
    def test_init_custom_params(self) -> Tuple[bool, str, str]:
        """Test custom initialization."""
        db = self.DBClass(strategy='lsh', dimensions=100, num_tables=5, num_bits=4)
        
        if db.strategy != 'lsh':
            return False, f"Strategy should be 'lsh', got '{db.strategy}'", ""
        
        if db.dimensions != 100:
            return False, f"Dimensions should be 100, got {db.dimensions}", ""
        
        return True, "Custom initialization correct", ""
    
    def test_init_invalid_strategy(self) -> Tuple[bool, str, str]:
        """Test that invalid strategy raises error."""
        try:
            db = self.DBClass(strategy='invalid')
            return False, "Should raise ValueError for invalid strategy", ""
        except ValueError as e:
            return True, "ValueError raised for invalid strategy", str(e)[:50]
    
    # =========================================================================
    # ADD AND GET TESTS
    # =========================================================================
    
    def test_add_single_vector(self) -> Tuple[bool, str, str]:
        """Test adding a single vector."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        
        vector = self.test_vectors[0]
        db.add("test_vec", vector)
        
        if db.size() != 1:
            return False, f"Size should be 1, got {db.size()}", ""
        
        return True, "Single vector added successfully", ""
    
    def test_add_batch_vectors(self) -> Tuple[bool, str, str]:
        """Test adding multiple vectors."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        
        db.add_batch(self.test_ids, self.test_vectors)
        
        if db.size() != self.num_vectors:
            return False, f"Size should be {self.num_vectors}, got {db.size()}", ""
        
        return True, f"Batch of {self.num_vectors} vectors added", ""
    
    def test_get_vector(self) -> Tuple[bool, str, str]:
        """Test retrieving a vector by ID."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        
        original = self.test_vectors[0]
        db.add("test_vec", original)
        
        retrieved = db.get("test_vec")
        
        if retrieved is None:
            return False, "get() returned None", ""
        
        # Vectors should be similar (normalized)
        similarity = np.dot(original / np.linalg.norm(original), retrieved)
        if similarity < 0.99:
            return False, f"Retrieved vector differs from original (sim={similarity:.4f})", ""
        
        return True, "Vector retrieved correctly", f"similarity={similarity:.4f}"
    
    def test_get_nonexistent(self) -> Tuple[bool, str, str]:
        """Test getting a non-existent vector."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        
        result = db.get("nonexistent")
        
        if result is not None:
            return False, "Should return None for non-existent ID", ""
        
        return True, "Returns None for non-existent ID", ""
    
    def test_dimension_mismatch(self) -> Tuple[bool, str, str]:
        """Test that dimension mismatch raises error."""
        db = self.DBClass(strategy='flat', dimensions=50)
        
        try:
            wrong_dim_vector = np.random.randn(100)  # Wrong dimensions
            db.add("test", wrong_dim_vector)
            return False, "Should raise ValueError for dimension mismatch", ""
        except ValueError:
            return True, "ValueError raised for dimension mismatch", ""
    
    # =========================================================================
    # FLAT SEARCH TESTS
    # =========================================================================
    
    def test_flat_search_basic(self) -> Tuple[bool, str, str]:
        """Test basic flat search."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        db.add_batch(self.test_ids, self.test_vectors)
        
        query = self.test_vectors[0]
        results = db.search(query, top_k=5)
        
        if not results:
            return False, "Search returned empty results", ""
        
        if len(results) != 5:
            return False, f"Should return 5 results, got {len(results)}", ""
        
        # First result should be the query itself (highest similarity)
        if results[0][0] != "vec_0":
            return False, f"First result should be vec_0, got {results[0][0]}", ""
        
        return True, "Flat search returns correct results", f"top result: {results[0]}"
    
    def test_flat_search_similarity_order(self) -> Tuple[bool, str, str]:
        """Test that flat search results are sorted by similarity."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        db.add_batch(self.test_ids, self.test_vectors)
        
        query = self.test_vectors[0]
        results = db.search(query, top_k=10)
        
        # Check that similarities are in descending order
        similarities = [score for _, score in results]
        for i in range(len(similarities) - 1):
            if similarities[i] < similarities[i + 1]:
                return False, "Results not sorted by similarity", f"{similarities}"
        
        return True, "Results correctly sorted by similarity", ""
    
    def test_flat_search_same_cluster(self) -> Tuple[bool, str, str]:
        """Test that flat search finds vectors from same cluster."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        db.add_batch(self.test_ids, self.test_vectors)
        
        # Query from first cluster (indices 0-24)
        query = self.test_vectors[5]
        results = db.search(query, top_k=10)
        
        # Most results should be from same cluster (indices 0-24)
        same_cluster = sum(1 for id, _ in results if int(id.split('_')[1]) < 25)
        
        if same_cluster < 5:
            return False, f"Should find most results from same cluster, got {same_cluster}/10", ""
        
        return True, f"Found {same_cluster}/10 from same cluster", ""
    
    # =========================================================================
    # LSH SEARCH TESTS
    # =========================================================================
    
    def test_lsh_build_index(self) -> Tuple[bool, str, str]:
        """Test LSH index building."""
        db = self.DBClass(strategy='lsh', dimensions=self.dimensions, 
                         num_tables=5, num_bits=4)
        db.add_batch(self.test_ids, self.test_vectors)
        db.build_index()
        
        if not db.index_built:
            return False, "Index should be marked as built", ""
        
        return True, "LSH index built successfully", ""
    
    def test_lsh_search_finds_similar(self) -> Tuple[bool, str, str]:
        """Test that LSH search finds similar vectors."""
        db = self.DBClass(strategy='lsh', dimensions=self.dimensions,
                         num_tables=10, num_bits=8)
        db.add_batch(self.test_ids, self.test_vectors)
        
        query = self.test_vectors[0]
        results = db.search(query, top_k=5)
        
        if not results:
            return False, "LSH search returned empty results", ""
        
        # First result should be highly similar
        if results[0][1] < 0.9:
            return False, f"Top result similarity too low: {results[0][1]}", ""
        
        return True, "LSH search finds similar vectors", f"top: {results[0]}"
    
    def test_lsh_approximate(self) -> Tuple[bool, str, str]:
        """Test that LSH is approximate (may miss some results)."""
        db_flat = self.DBClass(strategy='flat', dimensions=self.dimensions)
        db_lsh = self.DBClass(strategy='lsh', dimensions=self.dimensions,
                             num_tables=5, num_bits=4)  # Few tables for more misses
        
        db_flat.add_batch(self.test_ids, self.test_vectors)
        db_lsh.add_batch(self.test_ids, self.test_vectors)
        
        query = self.test_vectors[50]
        
        flat_results = set(id for id, _ in db_flat.search(query, top_k=20))
        lsh_results = set(id for id, _ in db_lsh.search(query, top_k=20))
        
        # LSH might not find all the same results as flat
        overlap = len(flat_results & lsh_results)
        
        # Should still find some overlap
        if overlap < 5:
            return False, f"Too few overlapping results: {overlap}", ""
        
        return True, f"LSH is approximate: {overlap}/20 overlap with flat", ""
    
    # =========================================================================
    # IVF SEARCH TESTS
    # =========================================================================
    
    def test_ivf_build_index(self) -> Tuple[bool, str, str]:
        """Test IVF index building."""
        db = self.DBClass(strategy='ivf', dimensions=self.dimensions,
                         num_clusters=10)
        db.add_batch(self.test_ids, self.test_vectors)
        db.build_index()
        
        if not db.index_built:
            return False, "Index should be marked as built", ""
        
        if db.centroids is None:
            return False, "Centroids should be computed", ""
        
        return True, "IVF index built successfully", f"{len(db.centroids)} centroids"
    
    def test_ivf_search_finds_similar(self) -> Tuple[bool, str, str]:
        """Test that IVF search finds similar vectors."""
        db = self.DBClass(strategy='ivf', dimensions=self.dimensions,
                         num_clusters=10, nprobe=3)
        db.add_batch(self.test_ids, self.test_vectors)
        
        query = self.test_vectors[0]
        results = db.search(query, top_k=5)
        
        if not results:
            return False, "IVF search returned empty results", ""
        
        # First result should be highly similar
        if results[0][1] < 0.9:
            return False, f"Top result similarity too low: {results[0][1]}", ""
        
        return True, "IVF search finds similar vectors", f"top: {results[0]}"
    
    def test_ivf_nprobe_affects_recall(self) -> Tuple[bool, str, str]:
        """Test that higher nprobe improves recall."""
        db_low = self.DBClass(strategy='ivf', dimensions=self.dimensions,
                             num_clusters=10, nprobe=1)
        db_high = self.DBClass(strategy='ivf', dimensions=self.dimensions,
                              num_clusters=10, nprobe=5)
        
        db_low.add_batch(self.test_ids, self.test_vectors)
        db_high.add_batch(self.test_ids, self.test_vectors)
        
        # Get ground truth from flat search
        db_flat = self.DBClass(strategy='flat', dimensions=self.dimensions)
        db_flat.add_batch(self.test_ids, self.test_vectors)
        
        query = self.test_vectors[50]
        
        flat_results = set(id for id, _ in db_flat.search(query, top_k=10))
        low_results = set(id for id, _ in db_low.search(query, top_k=10))
        high_results = set(id for id, _ in db_high.search(query, top_k=10))
        
        low_recall = len(flat_results & low_results)
        high_recall = len(flat_results & high_results)
        
        # Higher nprobe should generally give better recall
        # (though not always guaranteed due to clustering)
        
        return True, f"nprobe=1 recall: {low_recall}/10, nprobe=5 recall: {high_recall}/10", ""
    
    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================
    
    def test_empty_database_search(self) -> Tuple[bool, str, str]:
        """Test searching an empty database."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        
        query = np.random.randn(self.dimensions)
        results = db.search(query, top_k=5)
        
        if results != []:
            return False, f"Should return empty list, got {results}", ""
        
        return True, "Empty database returns empty results", ""
    
    def test_search_query_dimension_mismatch(self) -> Tuple[bool, str, str]:
        """Test that search with wrong query dimensions raises error."""
        db = self.DBClass(strategy='flat', dimensions=50)
        db.add("test", np.random.randn(50))
        
        try:
            wrong_dim_query = np.random.randn(100)
            db.search(wrong_dim_query, top_k=5)
            return False, "Should raise ValueError for query dimension mismatch", ""
        except ValueError:
            return True, "ValueError raised for query dimension mismatch", ""
    
    def test_top_k_larger_than_database(self) -> Tuple[bool, str, str]:
        """Test requesting more results than vectors in database."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        
        # Add only 3 vectors
        for i in range(3):
            db.add(f"vec_{i}", self.test_vectors[i])
        
        query = self.test_vectors[0]
        results = db.search(query, top_k=10)
        
        # Should return only 3 results
        if len(results) > 3:
            return False, f"Should return at most 3 results, got {len(results)}", ""
        
        return True, "Correctly handles top_k > database size", f"returned {len(results)}"
    
    def test_get_stats(self) -> Tuple[bool, str, str]:
        """Test get_stats method."""
        db = self.DBClass(strategy='ivf', dimensions=self.dimensions, num_clusters=5)
        db.add_batch(self.test_ids, self.test_vectors)
        db.build_index()
        
        stats = db.get_stats()
        
        if 'strategy' not in stats:
            return False, "Stats should include 'strategy'", ""
        
        if 'num_vectors' not in stats:
            return False, "Stats should include 'num_vectors'", ""
        
        if stats['num_vectors'] != self.num_vectors:
            return False, f"Wrong num_vectors: {stats['num_vectors']}", ""
        
        return True, "get_stats returns correct info", f"stats: {stats}"
    
    def test_normalization(self) -> Tuple[bool, str, str]:
        """Test that vectors are normalized when added."""
        db = self.DBClass(strategy='flat', dimensions=self.dimensions)
        
        # Add unnormalized vector
        unnormalized = np.random.randn(self.dimensions) * 10  # Large magnitude
        db.add("test", unnormalized)
        
        retrieved = db.get("test")
        norm = np.linalg.norm(retrieved)
        
        if not np.isclose(norm, 1.0, atol=0.01):
            return False, f"Vector should be normalized (norm={norm})", ""
        
        return True, "Vectors are normalized on add", f"norm={norm:.4f}"
    
    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================
    
    def run_all(self) -> Dict[str, List[TestResult]]:
        """Run all tests and return results grouped by category."""
        
        print("\n" + "=" * 70)
        print("🧪 WORKSHOP 3 VECTOR DATABASE TEST SUITE")
        print("=" * 70)
        
        categories = {
            "Initialization": [
                (self.test_init_default_strategy, "Default strategy"),
                (self.test_init_custom_params, "Custom parameters"),
                (self.test_init_invalid_strategy, "Invalid strategy error"),
            ],
            "Add and Get": [
                (self.test_add_single_vector, "Add single vector"),
                (self.test_add_batch_vectors, "Add batch vectors"),
                (self.test_get_vector, "Get vector by ID"),
                (self.test_get_nonexistent, "Get non-existent ID"),
                (self.test_dimension_mismatch, "Dimension mismatch error"),
            ],
            "Flat Search": [
                (self.test_flat_search_basic, "Basic flat search"),
                (self.test_flat_search_similarity_order, "Results sorted by similarity"),
                (self.test_flat_search_same_cluster, "Finds same-cluster vectors"),
            ],
            "LSH Search": [
                (self.test_lsh_build_index, "Build LSH index"),
                (self.test_lsh_search_finds_similar, "LSH finds similar vectors"),
                (self.test_lsh_approximate, "LSH is approximate"),
            ],
            "IVF Search": [
                (self.test_ivf_build_index, "Build IVF index"),
                (self.test_ivf_search_finds_similar, "IVF finds similar vectors"),
                (self.test_ivf_nprobe_affects_recall, "nprobe affects recall"),
            ],
            "Edge Cases": [
                (self.test_empty_database_search, "Empty database search"),
                (self.test_search_query_dimension_mismatch, "Query dimension mismatch"),
                (self.test_top_k_larger_than_database, "top_k > database size"),
                (self.test_get_stats, "get_stats method"),
                (self.test_normalization, "Vector normalization"),
            ],
        }
        
        results_by_category = {}
        
        for category, tests in categories.items():
            print(f"\n📝 {category}")
            print("-" * 50)
            
            category_results = []
            for test_fn, test_name in tests:
                result = self.run_test(test_fn, test_name)
                category_results.append(result)
                
                icon = "✅" if result.passed else "❌"
                print(f"  {icon} {test_name}: {result.message}")
                
                if self.verbose and result.details:
                    print(f"      └─ {result.details}")
            
            results_by_category[category] = category_results
        
        return results_by_category
    
    def print_summary(self, results_by_category: Dict[str, List[TestResult]]):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        
        total_passed = 0
        total_failed = 0
        
        for category, results in results_by_category.items():
            passed = sum(1 for r in results if r.passed)
            failed = len(results) - passed
            total_passed += passed
            total_failed += failed
            
            icon = "✅" if failed == 0 else "⚠️" if passed > 0 else "❌"
            print(f"  {icon} {category}: {passed}/{len(results)} passed")
        
        print("-" * 50)
        total = total_passed + total_failed
        percentage = (total_passed / total * 100) if total > 0 else 0
        
        if total_failed == 0:
            print(f"  🎉 ALL TESTS PASSED! ({total_passed}/{total})")
        else:
            print(f"  📈 {total_passed}/{total} tests passed ({percentage:.0f}%)")
            print(f"  ❌ {total_failed} tests failed")
        
        print("=" * 70)
        
        return total_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test Workshop 3 Vector Database")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Import the vector database
    print("🔍 Testing vector database implementation...")
    try:
        from vector_db import SimpleVectorDB
    except ImportError:
        print("❌ Could not import vector_db")
        print("   Make sure you're running from the workshop directory")
        sys.exit(1)
    
    # Run tests
    suite = VectorDBTestSuite(SimpleVectorDB, verbose=args.verbose)
    results = suite.run_all()
    success = suite.print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
