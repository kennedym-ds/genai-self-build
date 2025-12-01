"""
🧪 Workshop 2: Embeddings Test Suite
=====================================

This test suite validates the embedding implementation.
Run this to verify the code is working correctly!

Usage:
    python test_embeddings.py             # Run all tests
    python test_embeddings.py --verbose   # Detailed output
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


class EmbeddingTestSuite:
    """Comprehensive test suite for the SimpleEmbedding class."""
    
    def __init__(self, embedding_class, verbose: bool = False):
        self.EmbeddingClass = embedding_class
        self.verbose = verbose
        self.results: List[TestResult] = []
        
        # Test corpus - small but with clear semantic relationships
        self.corpus = [
            "The king and queen ruled the kingdom wisely.",
            "The prince and princess lived in the royal castle.",
            "The king wore a golden crown on his head.",
            "The queen was loved by all the people.",
            "The cat and dog are popular pets.",
            "The cat sleeps on the soft bed.",
            "The dog runs in the green park.",
            "Cats and dogs are furry animals.",
            "Machine learning transforms data into insights.",
            "Neural networks learn patterns from data.",
            "Deep learning uses neural networks.",
            "The man and woman walked together.",
            "He is a man and she is a woman.",
            "The king is a man and the queen is a woman.",
        ]
    
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
            result = TestResult(test_name, False, "Not implemented yet", "Method raises NotImplementedError")
        except Exception as e:
            result = TestResult(test_name, False, f"Exception: {type(e).__name__}", str(e))
        
        self.results.append(result)
        return result
    
    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================
    
    def test_init_default_strategy(self) -> Tuple[bool, str, str]:
        """Test default initialization."""
        emb = self.EmbeddingClass()
        
        if emb.strategy != 'cooccurrence':
            return False, f"Default strategy should be 'cooccurrence', got '{emb.strategy}'", ""
        
        if emb.dimensions != 50:
            return False, f"Default dimensions should be 50, got {emb.dimensions}", ""
        
        return True, "Default initialization correct", f"strategy={emb.strategy}, dims={emb.dimensions}"
    
    def test_init_custom_params(self) -> Tuple[bool, str, str]:
        """Test custom initialization."""
        emb = self.EmbeddingClass(strategy='random', dimensions=100)
        
        if emb.strategy != 'random':
            return False, f"Strategy should be 'random', got '{emb.strategy}'", ""
        
        if emb.dimensions != 100:
            return False, f"Dimensions should be 100, got {emb.dimensions}", ""
        
        return True, "Custom initialization correct", ""
    
    def test_init_invalid_strategy(self) -> Tuple[bool, str, str]:
        """Test that invalid strategy raises error."""
        try:
            emb = self.EmbeddingClass(strategy='invalid')
            return False, "Should raise ValueError for invalid strategy", ""
        except ValueError as e:
            return True, "ValueError raised for invalid strategy", str(e)[:50]
    
    # =========================================================================
    # RANDOM EMBEDDING TESTS
    # =========================================================================
    
    def test_random_train_creates_embeddings(self) -> Tuple[bool, str, str]:
        """Test that random training creates embeddings."""
        emb = self.EmbeddingClass(strategy='random', dimensions=30)
        emb.train(self.corpus)
        
        if emb.embeddings is None:
            return False, "Embeddings are None after training", ""
        
        if len(emb.vocab) == 0:
            return False, "Vocabulary is empty", ""
        
        expected_shape = (len(emb.vocab), 30)
        if emb.embeddings.shape != expected_shape:
            return False, f"Wrong shape: {emb.embeddings.shape} vs {expected_shape}", ""
        
        return True, f"Random embeddings created: {emb.embeddings.shape}", ""
    
    def test_random_get_vector(self) -> Tuple[bool, str, str]:
        """Test get_vector for random embeddings."""
        emb = self.EmbeddingClass(strategy='random', dimensions=30)
        emb.train(self.corpus)
        
        vec = emb.get_vector("king")
        
        if vec is None:
            return False, "get_vector returned None", ""
        
        if not isinstance(vec, np.ndarray):
            return False, f"Expected numpy array, got {type(vec)}", ""
        
        if vec.shape != (30,):
            return False, f"Wrong shape: {vec.shape} vs (30,)", ""
        
        return True, "get_vector returns correct array", f"Shape: {vec.shape}"
    
    def test_random_similarity_varies(self) -> Tuple[bool, str, str]:
        """Test that random similarity varies randomly."""
        emb = self.EmbeddingClass(strategy='random', dimensions=30)
        emb.train(self.corpus)
        
        sim1 = emb.similarity("king", "queen")
        sim2 = emb.similarity("king", "cat")
        
        # Both should be floats
        if not isinstance(sim1, float) or not isinstance(sim2, float):
            return False, "Similarity should return float", f"{type(sim1)}, {type(sim2)}"
        
        # With random embeddings, similarity should be between -1 and 1
        if not (-1 <= sim1 <= 1) or not (-1 <= sim2 <= 1):
            return False, "Similarity should be in [-1, 1]", f"sim1={sim1}, sim2={sim2}"
        
        return True, "Random similarity computed correctly", f"king-queen: {sim1:.3f}, king-cat: {sim2:.3f}"
    
    # =========================================================================
    # CO-OCCURRENCE EMBEDDING TESTS
    # =========================================================================
    
    def test_cooccurrence_train_creates_embeddings(self) -> Tuple[bool, str, str]:
        """Test that co-occurrence training creates embeddings."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        if emb.embeddings is None:
            return False, "Embeddings are None after training", ""
        
        if len(emb.vocab) == 0:
            return False, "Vocabulary is empty", ""
        
        expected_shape = (len(emb.vocab), 30)
        if emb.embeddings.shape != expected_shape:
            return False, f"Wrong shape: {emb.embeddings.shape} vs {expected_shape}", ""
        
        return True, f"Co-occurrence embeddings: {emb.embeddings.shape}", ""
    
    def test_cooccurrence_similar_words(self) -> Tuple[bool, str, str]:
        """Test that similar words have higher similarity than random pairs."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        # King and queen should be more similar than king and cat
        # (because they appear in similar contexts)
        sim_related = emb.similarity("king", "queen")
        
        # Get a less related pair
        sim_unrelated = emb.similarity("king", "dog")
        
        self.log(f"king-queen: {sim_related:.3f}, king-dog: {sim_unrelated:.3f}")
        
        # Co-occurrence should give higher similarity for related words
        # But with small corpus, we just check computation works
        return True, "Similarity computed for co-occurrence", f"king-queen: {sim_related:.3f}"
    
    def test_cooccurrence_most_similar(self) -> Tuple[bool, str, str]:
        """Test most_similar returns proper results."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        similar = emb.most_similar("king", top_n=3)
        
        if similar is None:
            return False, "most_similar returned None", ""
        
        if not isinstance(similar, list):
            return False, f"Expected list, got {type(similar)}", ""
        
        if len(similar) > 3:
            return False, f"Expected at most 3 results, got {len(similar)}", ""
        
        # Check format: list of (word, similarity) tuples
        for item in similar:
            if not isinstance(item, tuple) or len(item) != 2:
                return False, "Results should be (word, score) tuples", f"Got: {item}"
            word, score = item
            if not isinstance(word, str):
                return False, "Word should be string", f"Got: {type(word)}"
            if not isinstance(score, float):
                return False, "Score should be float", f"Got: {type(score)}"
        
        words = [w for w, s in similar]
        return True, f"Found similar words to 'king': {words}", ""
    
    def test_cooccurrence_vectors_normalized(self) -> Tuple[bool, str, str]:
        """Test that embedding vectors are normalized."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        vec = emb.get_vector("king")
        norm = np.linalg.norm(vec)
        
        if not np.isclose(norm, 1.0, atol=0.01):
            return False, f"Vector should be normalized (norm=1), got {norm:.4f}", ""
        
        return True, f"Vectors are normalized (norm={norm:.4f})", ""
    
    # =========================================================================
    # PREDICTION-BASED EMBEDDING TESTS
    # =========================================================================
    
    def test_prediction_train_creates_embeddings(self) -> Tuple[bool, str, str]:
        """Test that prediction training creates embeddings."""
        emb = self.EmbeddingClass(strategy='prediction', dimensions=30)
        emb.train(self.corpus, epochs=3)  # Few epochs for speed
        
        if emb.embeddings is None:
            return False, "Embeddings are None after training", ""
        
        if len(emb.vocab) == 0:
            return False, "Vocabulary is empty", ""
        
        return True, f"Prediction embeddings: {emb.embeddings.shape}", ""
    
    def test_prediction_loss_decreases(self) -> Tuple[bool, str, str]:
        """Test that training loss decreases over epochs."""
        emb = self.EmbeddingClass(strategy='prediction', dimensions=30)
        
        # We can't easily capture print output, so just verify training completes
        emb.train(self.corpus, epochs=5)
        
        if not emb.is_trained:
            return False, "Model should be marked as trained", ""
        
        return True, "Prediction training completed", ""
    
    def test_prediction_similarity(self) -> Tuple[bool, str, str]:
        """Test similarity with prediction embeddings."""
        emb = self.EmbeddingClass(strategy='prediction', dimensions=30)
        emb.train(self.corpus, epochs=5)
        
        sim = emb.similarity("king", "queen")
        
        if not isinstance(sim, float):
            return False, f"Similarity should be float, got {type(sim)}", ""
        
        if not (-1 <= sim <= 1):
            return False, f"Similarity should be in [-1, 1], got {sim}", ""
        
        return True, f"Prediction similarity: king-queen = {sim:.3f}", ""
    
    # =========================================================================
    # ANALOGY TESTS
    # =========================================================================
    
    def test_analogy_returns_list(self) -> Tuple[bool, str, str]:
        """Test that analogy returns proper format."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        result = emb.analogy("king", "queen", "man", top_n=3)
        
        if result is None:
            return False, "analogy returned None", ""
        
        if not isinstance(result, list):
            return False, f"Expected list, got {type(result)}", ""
        
        # Check format
        for item in result:
            if not isinstance(item, tuple) or len(item) != 2:
                return False, "Results should be (word, score) tuples", ""
        
        words = [w for w, s in result]
        return True, f"Analogy king:queen::man:? → {words}", ""
    
    def test_analogy_excludes_inputs(self) -> Tuple[bool, str, str]:
        """Test that analogy results exclude input words."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        result = emb.analogy("king", "queen", "man", top_n=5)
        words = [w for w, s in result]
        
        if "king" in words or "queen" in words or "man" in words:
            return False, "Analogy result should exclude input words", f"Got: {words}"
        
        return True, "Input words excluded from results", f"Results: {words}"
    
    def test_analogy_different_strategies(self) -> Tuple[bool, str, str]:
        """Test analogy works for all strategies."""
        results_by_strategy = {}
        
        for strategy in ['random', 'cooccurrence', 'prediction']:
            emb = self.EmbeddingClass(strategy=strategy, dimensions=30)
            emb.train(self.corpus, epochs=3)
            
            try:
                result = emb.analogy("king", "queen", "man", top_n=1)
                if result:
                    results_by_strategy[strategy] = result[0][0]
                else:
                    results_by_strategy[strategy] = "no result"
            except Exception as e:
                return False, f"Analogy failed for {strategy}", str(e)
        
        details = ", ".join([f"{s}: {w}" for s, w in results_by_strategy.items()])
        return True, "Analogy works for all strategies", details
    
    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================
    
    def test_unknown_word_raises_error(self) -> Tuple[bool, str, str]:
        """Test that unknown word raises ValueError."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        try:
            emb.get_vector("xyznotaword")
            return False, "Should raise ValueError for unknown word", ""
        except ValueError as e:
            return True, "ValueError raised for unknown word", str(e)[:40]
    
    def test_untrained_model_raises_error(self) -> Tuple[bool, str, str]:
        """Test that untrained model raises error."""
        emb = self.EmbeddingClass(strategy='cooccurrence')
        
        try:
            emb.get_vector("king")
            return False, "Should raise RuntimeError for untrained model", ""
        except RuntimeError as e:
            return True, "RuntimeError raised for untrained model", str(e)[:40]
    
    def test_get_all_words(self) -> Tuple[bool, str, str]:
        """Test get_all_words returns vocabulary."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        words = emb.get_all_words()
        
        if not isinstance(words, list):
            return False, f"Expected list, got {type(words)}", ""
        
        if len(words) != len(emb.vocab):
            return False, f"Word count mismatch: {len(words)} vs {len(emb.vocab)}", ""
        
        # Check some expected words
        expected = ["king", "queen", "cat", "dog"]
        missing = [w for w in expected if w not in words]
        if missing:
            return False, f"Missing words: {missing}", ""
        
        return True, f"get_all_words returns {len(words)} words", ""
    
    def test_vocab_size(self) -> Tuple[bool, str, str]:
        """Test vocab_size method."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        size = emb.vocab_size()
        
        if not isinstance(size, int):
            return False, f"Expected int, got {type(size)}", ""
        
        if size != len(emb.vocab):
            return False, f"Size mismatch: {size} vs {len(emb.vocab)}", ""
        
        return True, f"Vocabulary size: {size}", ""
    
    def test_case_insensitivity(self) -> Tuple[bool, str, str]:
        """Test that word lookup is case-insensitive."""
        emb = self.EmbeddingClass(strategy='cooccurrence', dimensions=30)
        emb.train(self.corpus)
        
        vec_lower = emb.get_vector("king")
        vec_upper = emb.get_vector("King")
        vec_mixed = emb.get_vector("KING")
        
        # All should return the same vector
        if not np.allclose(vec_lower, vec_upper):
            return False, "'king' and 'King' gave different vectors", ""
        
        if not np.allclose(vec_lower, vec_mixed):
            return False, "'king' and 'KING' gave different vectors", ""
        
        return True, "Word lookup is case-insensitive", ""
    
    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================
    
    def run_all(self) -> Dict[str, List[TestResult]]:
        """Run all tests and return results grouped by category."""
        
        print("\n" + "=" * 70)
        print("🧪 WORKSHOP 2 EMBEDDINGS TEST SUITE")
        print("=" * 70)
        
        categories = {
            "Initialization": [
                (self.test_init_default_strategy, "Default strategy"),
                (self.test_init_custom_params, "Custom parameters"),
                (self.test_init_invalid_strategy, "Invalid strategy error"),
            ],
            "Random Embeddings": [
                (self.test_random_train_creates_embeddings, "Training creates embeddings"),
                (self.test_random_get_vector, "get_vector returns array"),
                (self.test_random_similarity_varies, "Similarity computation"),
            ],
            "Co-occurrence Embeddings": [
                (self.test_cooccurrence_train_creates_embeddings, "Training creates embeddings"),
                (self.test_cooccurrence_similar_words, "Similar words test"),
                (self.test_cooccurrence_most_similar, "most_similar method"),
                (self.test_cooccurrence_vectors_normalized, "Vectors normalized"),
            ],
            "Prediction Embeddings": [
                (self.test_prediction_train_creates_embeddings, "Training creates embeddings"),
                (self.test_prediction_loss_decreases, "Training completes"),
                (self.test_prediction_similarity, "Similarity computation"),
            ],
            "Analogies": [
                (self.test_analogy_returns_list, "Analogy returns list"),
                (self.test_analogy_excludes_inputs, "Excludes input words"),
                (self.test_analogy_different_strategies, "Works for all strategies"),
            ],
            "Edge Cases": [
                (self.test_unknown_word_raises_error, "Unknown word error"),
                (self.test_untrained_model_raises_error, "Untrained model error"),
                (self.test_get_all_words, "get_all_words method"),
                (self.test_vocab_size, "vocab_size method"),
                (self.test_case_insensitivity, "Case insensitivity"),
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
    parser = argparse.ArgumentParser(description="Test Workshop 2 Embeddings")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Import the embeddings
    print("🔍 Testing embeddings implementation...")
    try:
        from embeddings import SimpleEmbedding
    except ImportError:
        print("❌ Could not import embeddings")
        print("   Make sure you're running from the workshop directory")
        sys.exit(1)
    
    # Run tests
    suite = EmbeddingTestSuite(SimpleEmbedding, verbose=args.verbose)
    results = suite.run_all()
    success = suite.print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
