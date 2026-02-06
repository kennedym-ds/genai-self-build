"""
🎯 Workshop 1: Text to Numbers (Tokenization)
=============================================

🛸 THE ALIEN ANALOGY:
Imagine an alien lands on Earth and needs to learn English. They don't know 
what "words" are—they just see a stream of symbols. But the alien notices patterns:

    "the" appears everywhere     → assign it one symbol: θ
    "ing" ends many words        → assign it one symbol: ω  
    "tion" is super common       → assign it one symbol: τ

After studying millions of texts, the alien builds a CODEBOOK of patterns.
Now they can read English in meaningful chunks, not letter by letter!

THIS IS EXACTLY WHAT TOKENIZATION DOES.
The "alien" is our algorithm. The "codebook" is our vocabulary.

A tokenizer converts text into numbers that neural networks can process.
This is the FIRST step in how ChatGPT, Claude, and all LLMs work!

THREE STRATEGIES:
1. Character-level: Like spelling out every letter on the phone (tedious!)
2. Word-level: Like a dictionary with a page number for each word (huge book!)
3. BPE (Byte Pair Encoding): Like shipping boxes—group items that ship together (smart!)

Usage:
    python tokenizer.py
"""

from collections import Counter
from typing import Dict, List, Tuple
import re


class SimpleTokenizer:
    """
    A multi-strategy tokenizer that can use character, word, or BPE tokenization.
    
    Example usage:
        tokenizer = SimpleTokenizer(strategy='word')
        tokenizer.train(["Hello world!", "Hello there!"])
        tokens = tokenizer.encode("Hello world!")  # [1, 2]
        text = tokenizer.decode([1, 2])            # "hello world"
    """
    
    def __init__(self, strategy: str = 'word', debug: bool = False):
        """
        Initialize the tokenizer.

        Args:
            strategy: One of 'char', 'word', or 'bpe'
            debug: If True, print detailed step-by-step information (for learning)
        """
        if strategy not in ['char', 'word', 'bpe']:
            raise ValueError(f"Strategy must be 'char', 'word', or 'bpe', got '{strategy}'")

        self.strategy = strategy
        self.debug = debug  # Enable "under the hood" visualization
        self.vocab: Dict[str, int] = {}          # token -> id
        self.inverse_vocab: Dict[int, str] = {}  # id -> token

        # Special tokens
        self.unk_token = '<UNK>'  # Unknown token
        self.pad_token = '<PAD>'  # Padding token

        # BPE-specific: list of merge rules learned during training
        self.merges: List[Tuple[str, str]] = []

        # 🔍 UNDER THE HOOD: Training statistics for visualization
        self.training_stats = {
            'total_chars': 0,
            'unique_chars': 0,
            'total_words': 0,
            'unique_words': 0,
            'bpe_iterations': 0,
            'compression_ratio': 1.0
        }
    
    # =========================================================================
    # PART 1: CHARACTER TOKENIZATION (Start here!)
    # =========================================================================
    
    def _train_char(self, corpus: List[str]) -> None:
        """
        Build character vocabulary from corpus.

        This is the simplest approach: each unique character gets an ID.

        Example:
            corpus = ["hello"]
            vocab = {'h': 0, 'e': 1, 'l': 2, 'o': 3}
        """
        # 🔍 UNDER THE HOOD: Show what we're processing
        if self.debug:
            print("\n" + "="*60)
            print("🔍 CHARACTER TOKENIZER TRAINING - UNDER THE HOOD")
            print("="*60)
            print(f"📥 Input: {len(corpus)} documents")
            for i, doc in enumerate(corpus[:3]):  # Show first 3
                print(f"   Doc {i+1}: '{doc[:50]}{'...' if len(doc) > 50 else ''}'")
            if len(corpus) > 3:
                print(f"   ... and {len(corpus) - 3} more")

        # Step 1: Combine all texts into one string
        all_text = ''.join(corpus)

        if self.debug:
            print(f"\n📊 Step 1: Combined all text")
            print(f"   Total characters: {len(all_text)}")
            print(f"   Sample: '{all_text[:100]}{'...' if len(all_text) > 100 else ''}'")

        # Step 2: Find all unique characters
        unique_chars = set(all_text)

        if self.debug:
            print(f"\n🎯 Step 2: Found unique characters")
            print(f"   Unique chars: {len(unique_chars)}")
            print(f"   Characters: {sorted(unique_chars)[:20]}{'...' if len(unique_chars) > 20 else ''}")

        # Step 3: Sort them for reproducibility
        sorted_chars = sorted(unique_chars)

        if self.debug:
            print(f"\n📋 Step 3: Sorted characters alphabetically")

        # Step 4 & 5: Assign IDs and store
        self.vocab = {char: idx for idx, char in enumerate(sorted_chars)}
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}

        # Update stats
        self.training_stats['total_chars'] = len(all_text)
        self.training_stats['unique_chars'] = len(unique_chars)

        if self.debug:
            print(f"\n✅ Step 4-5: Built vocabulary mapping")
            print(f"   Vocabulary size: {len(self.vocab)}")
            print(f"   Example mappings:")
            for char, idx in list(self.vocab.items())[:10]:
                print(f"      '{char}' → {idx}")
            if len(self.vocab) > 10:
                print(f"      ... and {len(self.vocab) - 10} more")
            print("="*60)
    
    def _encode_char(self, text: str) -> List[int]:
        """
        Encode text as a list of character IDs.

        Example:
            text = "hello"
            output = [0, 1, 2, 2, 3]  # h=0, e=1, l=2, o=3
        """
        # 🔍 UNDER THE HOOD: Show encoding process
        if self.debug:
            print("\n" + "-"*60)
            print(f"🔢 ENCODING: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print("-"*60)

        # For each character, look up its ID (skip unknown chars)
        result = []
        for i, char in enumerate(text):
            if char in self.vocab:
                token_id = self.vocab[char]
                result.append(token_id)
                if self.debug and i < 10:  # Show first 10
                    print(f"   '{char}' → {token_id}")
            elif self.debug and i < 10:
                print(f"   '{char}' → SKIPPED (not in vocab)")

        if self.debug:
            if len(text) > 10:
                print(f"   ... and {len(text) - 10} more characters")
            print(f"✅ Result: {len(result)} tokens")
            print("-"*60)

        return result
    
    def _decode_char(self, tokens: List[int]) -> str:
        """
        Decode character IDs back to text.
        
        Example:
            tokens = [0, 1, 2, 2, 3]
            output = "hello"
        """
        # Look up each ID and join into a string
        return ''.join(self.inverse_vocab.get(tok, '') for tok in tokens)
    
    # =========================================================================
    # PART 2: WORD TOKENIZATION (After completing Part 1)
    # =========================================================================
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Split text into words.
        
        We'll use a simple approach: lowercase and split on non-alphanumeric.
        
        Example:
            text = "Hello, World!"
            output = ["hello", "world"]
        """
        # Lowercase and find all word sequences
        return re.findall(r'\b\w+\b', text.lower())
    
    def _train_word(self, corpus: List[str], vocab_size: int = 10000) -> None:
        """
        Build word vocabulary from corpus.

        We want the most COMMON words in our vocabulary.

        Example:
            corpus = ["hello world", "hello there"]
            vocab = {'<UNK>': 0, '<PAD>': 1, 'hello': 2, 'world': 3, 'there': 4}
        """
        # 🔍 UNDER THE HOOD: Show training process
        if self.debug:
            print("\n" + "="*60)
            print("🔍 WORD TOKENIZER TRAINING - UNDER THE HOOD")
            print("="*60)
            print(f"📥 Input: {len(corpus)} documents")
            print(f"🎯 Target vocabulary size: {vocab_size}")

        # Step 1: Tokenize all texts into words
        all_words = []
        for text in corpus:
            all_words.extend(self._tokenize_text(text))

        if self.debug:
            print(f"\n📊 Step 1: Tokenized all documents")
            print(f"   Total words: {len(all_words)}")
            print(f"   First 10 words: {all_words[:10]}")

        # Step 2: Count word frequencies
        word_counts = Counter(all_words)

        if self.debug:
            print(f"\n🎯 Step 2: Counted word frequencies")
            print(f"   Unique words: {len(word_counts)}")
            print(f"   Top 10 most common:")
            for word, count in word_counts.most_common(10):
                print(f"      '{word}': {count} times")

        # Step 3: Keep top (vocab_size - 2) words
        most_common = word_counts.most_common(vocab_size - 2)

        if self.debug:
            print(f"\n📋 Step 3: Selected top {len(most_common)} words")
            print(f"   (Reserving 2 slots for special tokens)")

        # Step 4: Add special tokens first
        self.vocab = {self.unk_token: 0, self.pad_token: 1}
        self.inverse_vocab = {0: self.unk_token, 1: self.pad_token}

        if self.debug:
            print(f"\n🏷️  Step 4: Added special tokens")
            print(f"   {self.unk_token} → 0 (for unknown words)")
            print(f"   {self.pad_token} → 1 (for padding sequences)")

        # Step 5 & 6: Assign IDs to most common words
        for idx, (word, count) in enumerate(most_common, start=2):
            self.vocab[word] = idx
            self.inverse_vocab[idx] = word

        # Update stats
        self.training_stats['total_words'] = len(all_words)
        self.training_stats['unique_words'] = len(word_counts)

        if self.debug:
            print(f"\n✅ Step 5-6: Built final vocabulary")
            print(f"   Final vocabulary size: {len(self.vocab)}")
            print(f"   Coverage: {len(most_common)}/{len(word_counts)} unique words")
            coverage_pct = len(most_common) / len(word_counts) * 100
            print(f"   ({coverage_pct:.1f}% of unique words)")
            print("="*60)
    
    def _encode_word(self, text: str) -> List[int]:
        """
        Encode text as a list of word IDs.
        
        Unknown words should map to <UNK> (ID 0).
        
        Example:
            text = "hello world"
            output = [2, 3]  # assuming hello=2, world=3
        """
        # Tokenize and look up each word (0 = UNK for unknown)
        words = self._tokenize_text(text)
        return [self.vocab.get(word, 0) for word in words]
    
    def _decode_word(self, tokens: List[int]) -> str:
        """
        Decode word IDs back to text.
        
        Example:
            tokens = [2, 3]
            output = "hello world"
        """
        # Look up each ID and join with spaces
        words = [self.inverse_vocab.get(tok, self.unk_token) for tok in tokens]
        return ' '.join(words)
    
    # =========================================================================
    # PART 3: BPE TOKENIZATION (Stretch Goal!)
    # =========================================================================
    
    def _get_word_freqs(self, corpus: List[str]) -> Dict[tuple, int]:
        """
        Convert corpus into word frequencies with characters separated.
        
        Example:
            corpus = ["hello hello world"]
            output = {
                ('h', 'e', 'l', 'l', 'o', '</w>'): 2,  # hello appears twice
                ('w', 'o', 'r', 'l', 'd', '</w>'): 1,
            }
        
        Note: </w> marks end of word (helps with reconstruction)
        """
        word_freqs: Dict[tuple, int] = {}
        for text in corpus:
            words = self._tokenize_text(text)
            for word in words:
                # Convert word to tuple of characters + end marker
                chars = tuple(list(word) + ['</w>'])
                word_freqs[chars] = word_freqs.get(chars, 0) + 1
        return word_freqs
    
    def _get_pair_freqs(self, word_freqs: Dict[tuple, int]) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of adjacent character pairs across all words.
        
        Example:
            word_freqs = {('h', 'e', 'l', 'l', 'o', '</w>'): 2}
            output = {
                ('h', 'e'): 2,
                ('e', 'l'): 2,
                ('l', 'l'): 2,
                ('l', 'o'): 2,
                ('o', '</w>'): 2,
            }
        """
        pair_freqs: Dict[Tuple[str, str], int] = {}
        
        for word, freq in word_freqs.items():
            # Look at each adjacent pair
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        
        return pair_freqs
    
    def _merge_pair(self, word_freqs: Dict[tuple, int], 
                    pair: Tuple[str, str]) -> Dict[tuple, int]:
        """
        Merge a pair of characters in all words.
        
        Example:
            word_freqs = {('h', 'e', 'l', 'l', 'o', '</w>'): 2}
            pair = ('l', 'l')
            output = {('h', 'e', 'll', 'o', '</w>'): 2}
        """
        new_word_freqs: Dict[tuple, int] = {}
        merged_token = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                # Check if current position matches the pair
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged_token)
                    i += 2  # Skip both characters
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def _train_bpe(self, corpus: List[str], vocab_size: int = 1000) -> None:
        """
        Train BPE tokenizer by iteratively merging most frequent pairs.
        
        Algorithm:
            1. Start with characters as initial vocabulary
            2. Count all adjacent pairs
            3. Merge the most frequent pair
            4. Repeat until vocab_size reached
        """
        # Step 1: Get word frequencies
        word_freqs = self._get_word_freqs(corpus)
        
        # Step 2: Build initial character vocabulary
        self.vocab = {'</w>': 0}
        for word in word_freqs.keys():
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        # Step 3: Iteratively merge most frequent pairs
        while len(self.vocab) < vocab_size:
            pair_freqs = self._get_pair_freqs(word_freqs)
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs.keys(), key=lambda p: pair_freqs[p])
            
            # Merge that pair
            word_freqs = self._merge_pair(word_freqs, best_pair)
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
            
            # Record the merge
            self.merges.append(best_pair)
        
        # Create inverse vocabulary
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word."""
        tokens = list(word) + ['</w>']
        
        # Apply each merge in order
        for pair in self.merges:
            merged = pair[0] + pair[1]
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                else:
                    i += 1
        
        return tokens
    
    def _encode_bpe(self, text: str) -> List[int]:
        """Encode text using learned BPE merges."""
        words = self._tokenize_text(text)
        token_ids = []
        
        for word in words:
            word_tokens = self._apply_bpe(word)
            for token in word_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Fall back to characters
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
        
        return token_ids
    
    def _decode_bpe(self, tokens: List[int]) -> str:
        """Decode BPE token IDs back to text."""
        result = []
        current_word = []
        
        for tok in tokens:
            token = self.inverse_vocab.get(tok, '')
            
            if token == '</w>':
                result.append(''.join(current_word))
                current_word = []
            elif token.endswith('</w>'):
                current_word.append(token[:-4])
                result.append(''.join(current_word))
                current_word = []
            else:
                current_word.append(token)
        
        if current_word:
            result.append(''.join(current_word))
        
        return ' '.join(result)
    
    # =========================================================================
    # PUBLIC METHODS (These call your implementations above)
    # =========================================================================
    
    def train(self, corpus: List[str], vocab_size: int = 10000) -> None:
        """Train the tokenizer on a corpus of texts."""
        if self.strategy == 'char':
            self._train_char(corpus)
        elif self.strategy == 'word':
            self._train_word(corpus, vocab_size)
        elif self.strategy == 'bpe':
            self._train_bpe(corpus, vocab_size)
        
        print(f"✅ Trained {self.strategy} tokenizer with {len(self.vocab)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs."""
        if self.strategy == 'char':
            return self._encode_char(text) or []
        elif self.strategy == 'word':
            return self._encode_word(text) or []
        elif self.strategy == 'bpe':
            return self._encode_bpe(text) or []
        return []
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back into text."""
        if self.strategy == 'char':
            return self._decode_char(tokens) or ''
        elif self.strategy == 'word':
            return self._decode_word(tokens) or ''
        elif self.strategy == 'bpe':
            return self._decode_bpe(tokens) or ''
        return ''
    
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)

    def get_stats(self) -> dict:
        """
        🔍 UNDER THE HOOD: Get detailed statistics about the tokenizer.

        Returns a dictionary with training metrics and performance stats.
        Useful for understanding what's happening inside the tokenizer!
        """
        stats = self.training_stats.copy()
        stats['vocab_size'] = len(self.vocab)
        stats['strategy'] = self.strategy

        if self.strategy == 'bpe':
            stats['num_merges'] = len(self.merges)

        # Calculate compression ratio if we have the data
        if stats['total_chars'] > 0 and stats.get('total_words', 0) > 0:
            if self.strategy == 'word':
                # For word tokenizer: compare words to chars
                stats['compression_ratio'] = stats['total_chars'] / stats['total_words']
            elif self.strategy == 'char':
                # For char tokenizer: no compression (1:1)
                stats['compression_ratio'] = 1.0

        return stats


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_tokenizer():
    """Test all three tokenization strategies."""
    
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we build software.",
        "Python is a popular programming language.",
        "Tokenization is the first step in NLP.",
        "The fox is quick and clever.",
        "Learning to code is fun and rewarding.",
    ]
    
    test_text = "The quick fox is learning."
    
    print("=" * 60)
    print("🧪 TOKENIZER TEST SUITE")
    print("=" * 60)
    
    # Test 1: Character Tokenizer
    print("\n📝 TEST 1: Character Tokenizer")
    print("-" * 40)
    try:
        char_tok = SimpleTokenizer(strategy='char')
        char_tok.train(corpus)
        
        encoded = char_tok.encode(test_text)
        decoded = char_tok.decode(encoded)
        
        print(f"Vocabulary size: {char_tok.vocab_size()}")
        print(f"Original:  '{test_text}'")
        print(f"Encoded:   {encoded[:15]}... (length: {len(encoded)})")
        print(f"Decoded:   '{decoded}'")
        print(f"Roundtrip: {'✅ PASS' if decoded == test_text else '❌ FAIL'}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Word Tokenizer
    print("\n📝 TEST 2: Word Tokenizer")
    print("-" * 40)
    try:
        word_tok = SimpleTokenizer(strategy='word')
        word_tok.train(corpus, vocab_size=100)
        
        encoded = word_tok.encode(test_text)
        decoded = word_tok.decode(encoded)
        
        print(f"Vocabulary size: {word_tok.vocab_size()}")
        print(f"Original:  '{test_text}'")
        print(f"Encoded:   {encoded}")
        print(f"Decoded:   '{decoded}'")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: BPE Tokenizer
    print("\n📝 TEST 3: BPE Tokenizer (Stretch Goal)")
    print("-" * 40)
    try:
        bpe_tok = SimpleTokenizer(strategy='bpe')
        bpe_tok.train(corpus, vocab_size=100)
        
        encoded = bpe_tok.encode(test_text)
        decoded = bpe_tok.decode(encoded)
        
        print(f"Vocabulary size: {bpe_tok.vocab_size()}")
        print(f"Merges learned: {len(bpe_tok.merges)}")
        print(f"Encoded:   {encoded}")
        print(f"Decoded:   '{decoded}'")
    except Exception as e:
        print(f"⚠️  BPE not implemented yet: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_tokenizer()
