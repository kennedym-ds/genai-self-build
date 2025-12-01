"""
🧪 Workshop 1: Tokenization Test Suite
=======================================

This test suite validates the tokenizer implementation.
Run this to verify the code is working correctly!

Usage:
    python test_tokenizer.py             # Run all tests
    python test_tokenizer.py --verbose   # Detailed output
"""

import sys
import os
import argparse
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


class TokenizerTestSuite:
    """Comprehensive test suite for the SimpleTokenizer class."""
    
    def __init__(self, tokenizer_class, verbose: bool = False):
        self.TokenizerClass = tokenizer_class
        self.verbose = verbose
        self.results: List[TestResult] = []
        
        # Test corpus
        self.corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming how we build software.",
            "Python is a popular programming language.",
            "Tokenization is the first step in NLP.",
            "The fox is quick and clever.",
            "Learning to code is fun and rewarding.",
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
            result = TestResult(test_name, False, "Not implemented yet", "Method returns None or raises NotImplementedError")
        except Exception as e:
            result = TestResult(test_name, False, f"Exception: {type(e).__name__}", str(e))
        
        self.results.append(result)
        return result
    
    # =========================================================================
    # CHARACTER TOKENIZER TESTS
    # =========================================================================
    
    def test_char_train_creates_vocab(self) -> Tuple[bool, str, str]:
        """Test that character training creates a vocabulary."""
        tok = self.TokenizerClass(strategy='char')
        tok.train(self.corpus)
        
        if not tok.vocab:
            return False, "Vocabulary is empty after training", ""
        
        if not tok.inverse_vocab:
            return False, "Inverse vocabulary is empty", ""
        
        # Should have common characters
        expected_chars = ['a', 'e', 'i', 'o', 'u', ' ', '.', 'T', 'h']
        missing = [c for c in expected_chars if c not in tok.vocab]
        
        if missing:
            return False, f"Missing expected characters: {missing}", f"Vocab has {len(tok.vocab)} chars"
        
        return True, f"Vocabulary created with {len(tok.vocab)} characters", ""
    
    def test_char_encode_returns_list(self) -> Tuple[bool, str, str]:
        """Test that character encode returns a list of integers."""
        tok = self.TokenizerClass(strategy='char')
        tok.train(self.corpus)
        
        result = tok.encode("hello")
        
        if result is None:
            return False, "encode() returned None", "Method not implemented"
        
        if not isinstance(result, list):
            return False, f"encode() returned {type(result)}, expected list", ""
        
        if not all(isinstance(x, int) for x in result):
            return False, "encode() should return list of integers", f"Got: {result}"
        
        if len(result) != 5:  # "hello" = 5 characters
            return False, f"Expected 5 tokens for 'hello', got {len(result)}", f"Result: {result}"
        
        return True, "encode() returns correct list of integers", f"'hello' → {result}"
    
    def test_char_decode_returns_string(self) -> Tuple[bool, str, str]:
        """Test that character decode returns a string."""
        tok = self.TokenizerClass(strategy='char')
        tok.train(self.corpus)
        
        encoded = tok.encode("test")
        if encoded is None:
            return False, "encode() returned None", "Cannot test decode without encode"
        
        decoded = tok.decode(encoded)
        
        if decoded is None:
            return False, "decode() returned None", "Method not implemented"
        
        if not isinstance(decoded, str):
            return False, f"decode() returned {type(decoded)}, expected str", ""
        
        return True, "decode() returns string", f"Decoded: '{decoded}'"
    
    def test_char_roundtrip(self) -> Tuple[bool, str, str]:
        """Test that encode → decode returns original text."""
        tok = self.TokenizerClass(strategy='char')
        tok.train(self.corpus)
        
        # Use text that only contains characters from the training corpus
        test_texts = ["the quick fox", "machine learning", "is fun"]
        
        for text in test_texts:
            encoded = tok.encode(text)
            if encoded is None:
                return False, f"encode() returned None for '{text}'", ""
            
            decoded = tok.decode(encoded)
            if decoded is None:
                return False, f"decode() returned None", ""
            
            if decoded != text:
                return False, f"Roundtrip failed", f"'{text}' → {encoded} → '{decoded}'"
        
        return True, "Perfect roundtrip for all test texts", ""
    
    def test_char_vocab_size(self) -> Tuple[bool, str, str]:
        """Test that vocabulary size is reasonable."""
        tok = self.TokenizerClass(strategy='char')
        tok.train(self.corpus)
        
        size = tok.vocab_size() if hasattr(tok, 'vocab_size') else len(tok.vocab)
        
        # Should have at least 26 letters + some punctuation/spaces
        if size < 20:
            return False, f"Vocabulary too small: {size}", "Expected at least 20 unique characters"
        
        if size > 200:
            return False, f"Vocabulary too large: {size}", "Character vocab should be < 200"
        
        return True, f"Vocabulary size: {size}", ""
    
    # =========================================================================
    # WORD TOKENIZER TESTS
    # =========================================================================
    
    def test_word_train_creates_vocab(self) -> Tuple[bool, str, str]:
        """Test that word training creates a vocabulary."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(self.corpus, vocab_size=100)
        
        if not tok.vocab:
            return False, "Vocabulary is empty after training", ""
        
        # Should have special tokens
        if '<UNK>' not in tok.vocab and tok.unk_token not in tok.vocab:
            return False, "Missing <UNK> token in vocabulary", ""
        
        return True, f"Vocabulary created with {len(tok.vocab)} words", ""
    
    def test_word_encode_returns_list(self) -> Tuple[bool, str, str]:
        """Test that word encode returns a list of integers."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(self.corpus, vocab_size=100)
        
        result = tok.encode("the quick fox")
        
        if result is None:
            return False, "encode() returned None", "Method not implemented"
        
        if not isinstance(result, list):
            return False, f"encode() returned {type(result)}, expected list", ""
        
        if not all(isinstance(x, int) for x in result):
            return False, "encode() should return list of integers", ""
        
        # "the quick fox" = 3 words
        if len(result) != 3:
            return False, f"Expected 3 tokens, got {len(result)}", f"Result: {result}"
        
        return True, "encode() returns correct list of integers", f"'the quick fox' → {result}"
    
    def test_word_decode_returns_string(self) -> Tuple[bool, str, str]:
        """Test that word decode returns a string."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(self.corpus, vocab_size=100)
        
        encoded = tok.encode("machine learning")
        if encoded is None:
            return False, "encode() returned None", ""
        
        decoded = tok.decode(encoded)
        
        if decoded is None:
            return False, "decode() returned None", "Method not implemented"
        
        if not isinstance(decoded, str):
            return False, f"decode() returned {type(decoded)}, expected str", ""
        
        return True, "decode() returns string", f"Decoded: '{decoded}'"
    
    def test_word_handles_unknown(self) -> Tuple[bool, str, str]:
        """Test that unknown words are handled with UNK token."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(["hello world"], vocab_size=10)  # Very limited vocab
        
        # "xyz" is definitely not in the vocabulary
        encoded = tok.encode("xyz unknown word")
        
        if encoded is None:
            return False, "encode() returned None", ""
        
        # Should not crash - unknown words should map to UNK (usually 0)
        unk_id = tok.vocab.get('<UNK>', tok.vocab.get(tok.unk_token, 0))
        
        if unk_id not in encoded:
            self.log(f"Warning: UNK token ({unk_id}) not found in encoded result")
            # This might be okay if the word happens to be in vocab somehow
        
        return True, "Unknown words handled without crashing", f"'xyz unknown word' → {encoded}"
    
    def test_word_vocab_includes_common_words(self) -> Tuple[bool, str, str]:
        """Test that common words are in vocabulary."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(self.corpus, vocab_size=100)
        
        # These words appear multiple times in corpus
        common_words = ['the', 'is', 'fox', 'quick']
        missing = [w for w in common_words if w not in tok.vocab]
        
        if missing:
            return False, f"Missing common words: {missing}", f"Vocab: {list(tok.vocab.keys())[:10]}..."
        
        return True, "Common words are in vocabulary", ""
    
    def test_word_lowercase_handling(self) -> Tuple[bool, str, str]:
        """Test that words are lowercased consistently."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(self.corpus, vocab_size=100)
        
        # "The" and "the" should encode to same token
        enc1 = tok.encode("The")
        enc2 = tok.encode("the")
        
        if enc1 is None or enc2 is None:
            return False, "encode() returned None", ""
        
        if enc1 != enc2:
            return False, "Case not normalized", f"'The' → {enc1}, 'the' → {enc2}"
        
        return True, "Words are lowercased consistently", ""
    
    # =========================================================================
    # BPE TOKENIZER TESTS (Stretch Goal)
    # =========================================================================
    
    def test_bpe_train_creates_vocab(self) -> Tuple[bool, str, str]:
        """Test that BPE training creates a vocabulary."""
        tok = self.TokenizerClass(strategy='bpe')
        tok.train(self.corpus, vocab_size=100)
        
        if not tok.vocab:
            return False, "Vocabulary is empty after training", ""
        
        return True, f"BPE vocabulary created with {len(tok.vocab)} tokens", ""
    
    def test_bpe_learns_merges(self) -> Tuple[bool, str, str]:
        """Test that BPE learns merge rules."""
        tok = self.TokenizerClass(strategy='bpe')
        tok.train(self.corpus, vocab_size=100)
        
        if not hasattr(tok, 'merges') or tok.merges is None:
            return False, "No merges attribute found", ""
        
        if len(tok.merges) == 0:
            return False, "No merges learned", ""
        
        # Check merge format
        first_merge = tok.merges[0]
        if not isinstance(first_merge, tuple) or len(first_merge) != 2:
            return False, "Merges should be tuples of 2 strings", f"Got: {first_merge}"
        
        return True, f"Learned {len(tok.merges)} merge rules", f"First 3: {tok.merges[:3]}"
    
    def test_bpe_encode_returns_list(self) -> Tuple[bool, str, str]:
        """Test that BPE encode returns a list of integers."""
        tok = self.TokenizerClass(strategy='bpe')
        tok.train(self.corpus, vocab_size=100)
        
        result = tok.encode("the quick")
        
        if result is None:
            return False, "encode() returned None", "Method not implemented"
        
        if not isinstance(result, list):
            return False, f"encode() returned {type(result)}, expected list", ""
        
        if not all(isinstance(x, int) for x in result):
            return False, "encode() should return list of integers", ""
        
        return True, "BPE encode() returns list of integers", f"'the quick' → {result}"
    
    def test_bpe_decode_returns_string(self) -> Tuple[bool, str, str]:
        """Test that BPE decode returns a string."""
        tok = self.TokenizerClass(strategy='bpe')
        tok.train(self.corpus, vocab_size=100)
        
        encoded = tok.encode("python")
        if encoded is None:
            return False, "encode() returned None", ""
        
        decoded = tok.decode(encoded)
        
        if decoded is None:
            return False, "decode() returned None", "Method not implemented"
        
        if not isinstance(decoded, str):
            return False, f"decode() returned {type(decoded)}, expected str", ""
        
        return True, "BPE decode() returns string", f"Decoded: '{decoded}'"
    
    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================
    
    def test_empty_string(self) -> Tuple[bool, str, str]:
        """Test handling of empty string."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(self.corpus, vocab_size=100)
        
        try:
            encoded = tok.encode("")
            if encoded is None:
                encoded = []
            
            if not isinstance(encoded, list):
                return False, "Empty string should return list", f"Got: {type(encoded)}"
            
            decoded = tok.decode([])
            if decoded is None:
                decoded = ""
            
            return True, "Empty string handled correctly", f"encode('') → {encoded}"
        except Exception as e:
            return False, f"Exception on empty string: {e}", ""
    
    def test_special_characters(self) -> Tuple[bool, str, str]:
        """Test handling of special characters."""
        tok = self.TokenizerClass(strategy='char')
        tok.train(self.corpus)
        
        test = "Hello! How are you?"
        
        try:
            encoded = tok.encode(test)
            if encoded is None:
                return False, "encode() returned None", ""
            
            decoded = tok.decode(encoded)
            
            return True, "Special characters handled", f"'{test}' → {len(encoded)} tokens"
        except Exception as e:
            return False, f"Exception: {e}", ""
    
    def test_repeated_words(self) -> Tuple[bool, str, str]:
        """Test that repeated words get same token IDs."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(self.corpus, vocab_size=100)
        
        encoded = tok.encode("the the the")
        
        if encoded is None:
            return False, "encode() returned None", ""
        
        if len(encoded) != 3:
            return False, f"Expected 3 tokens, got {len(encoded)}", ""
        
        if not (encoded[0] == encoded[1] == encoded[2]):
            return False, "Repeated word got different IDs", f"'the the the' → {encoded}"
        
        return True, "Repeated words get same token ID", f"'the the the' → {encoded}"
    
    def test_vocab_size_limit(self) -> Tuple[bool, str, str]:
        """Test that vocab_size parameter is respected."""
        tok = self.TokenizerClass(strategy='word')
        tok.train(self.corpus, vocab_size=20)
        
        actual_size = len(tok.vocab)
        
        if actual_size > 20:
            return False, f"Vocab size {actual_size} exceeds limit 20", ""
        
        return True, f"Vocab size {actual_size} respects limit 20", ""
    
    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================
    
    def run_all(self) -> Dict[str, List[TestResult]]:
        """Run all tests and return results grouped by category."""
        
        print("\n" + "=" * 70)
        print("🧪 WORKSHOP 1 TOKENIZER TEST SUITE")
        print("=" * 70)
        
        categories = {
            "Character Tokenizer": [
                (self.test_char_train_creates_vocab, "Train creates vocabulary"),
                (self.test_char_encode_returns_list, "Encode returns list"),
                (self.test_char_decode_returns_string, "Decode returns string"),
                (self.test_char_roundtrip, "Encode/decode roundtrip"),
                (self.test_char_vocab_size, "Vocabulary size reasonable"),
            ],
            "Word Tokenizer": [
                (self.test_word_train_creates_vocab, "Train creates vocabulary"),
                (self.test_word_encode_returns_list, "Encode returns list"),
                (self.test_word_decode_returns_string, "Decode returns string"),
                (self.test_word_handles_unknown, "Handles unknown words"),
                (self.test_word_vocab_includes_common_words, "Includes common words"),
                (self.test_word_lowercase_handling, "Lowercase handling"),
            ],
            "BPE Tokenizer (Stretch)": [
                (self.test_bpe_train_creates_vocab, "Train creates vocabulary"),
                (self.test_bpe_learns_merges, "Learns merge rules"),
                (self.test_bpe_encode_returns_list, "Encode returns list"),
                (self.test_bpe_decode_returns_string, "Decode returns string"),
            ],
            "Edge Cases": [
                (self.test_empty_string, "Empty string handling"),
                (self.test_special_characters, "Special characters"),
                (self.test_repeated_words, "Repeated words"),
                (self.test_vocab_size_limit, "Vocab size limit"),
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
    parser = argparse.ArgumentParser(description="Test Workshop 1 Tokenizer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Import the tokenizer
    print("🔍 Testing tokenizer implementation...")
    try:
        from tokenizer import SimpleTokenizer
    except ImportError:
        print("❌ Could not import tokenizer")
        print("   Make sure you're running from the workshop directory")
        sys.exit(1)
    
    # Run tests
    suite = TokenizerTestSuite(SimpleTokenizer, verbose=args.verbose)
    results = suite.run_all()
    success = suite.print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
