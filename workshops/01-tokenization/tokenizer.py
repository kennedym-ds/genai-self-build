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
    
    def __init__(self, strategy: str = 'word'):
        """
        Initialize the tokenizer.
        
        Args:
            strategy: One of 'char', 'word', or 'bpe'
        """
        if strategy not in ['char', 'word', 'bpe']:
            raise ValueError(f"Strategy must be 'char', 'word', or 'bpe', got '{strategy}'")
        
        self.strategy = strategy
        self.vocab: Dict[str, int] = {}          # token -> id
        self.inverse_vocab: Dict[int, str] = {}  # id -> token
        
        # Special tokens
        self.unk_token = '<UNK>'  # Unknown token
        self.pad_token = '<PAD>'  # Padding token
        
        # BPE-specific: list of merge rules learned during training
        self.merges: List[Tuple[str, str]] = []
    
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
        # Step 1: Combine all texts into one string
        all_text = ''.join(corpus)
        
        # Step 2: Find all unique characters
        unique_chars = set(all_text)
        
        # Step 3: Sort them for reproducibility
        sorted_chars = sorted(unique_chars)
        
        # Step 4 & 5: Assign IDs and store
        self.vocab = {char: idx for idx, char in enumerate(sorted_chars)}
        self.inverse_vocab = {idx: char for char, idx in self.vocab.items()}
    
    def _encode_char(self, text: str) -> List[int]:
        """
        Encode text as a list of character IDs.
        
        Example:
            text = "hello"
            output = [0, 1, 2, 2, 3]  # h=0, e=1, l=2, o=3
        """
        # For each character, look up its ID (skip unknown chars)
        return [self.vocab[char] for char in text if char in self.vocab]
    
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
        # Step 1: Tokenize all texts into words
        all_words = []
        for text in corpus:
            all_words.extend(self._tokenize_text(text))
        
        # Step 2: Count word frequencies
        word_counts = Counter(all_words)
        
        # Step 3: Keep top (vocab_size - 2) words
        most_common = word_counts.most_common(vocab_size - 2)
        
        # Step 4: Add special tokens first
        self.vocab = {self.unk_token: 0, self.pad_token: 1}
        self.inverse_vocab = {0: self.unk_token, 1: self.pad_token}
        
        # Step 5 & 6: Assign IDs to most common words
        for idx, (word, _) in enumerate(most_common, start=2):
            self.vocab[word] = idx
            self.inverse_vocab[idx] = word
    
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
