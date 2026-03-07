#!/usr/bin/env python3
"""
🔍 Under the Hood Demo - See How Tokenization Really Works!
===========================================================

This script demonstrates the debug mode that shows step-by-step
what's happening inside the tokenizer.

Usage:
    python under_the_hood_demo.py
"""

from tokenizer import SimpleTokenizer

# Sample corpus for training
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming how we build software.",
    "Python is a popular programming language.",
    "Tokenization is the first step in natural language processing.",
]

print("=" * 70)
print("🔍 UNDER THE HOOD: TOKENIZATION DEMO")
print("=" * 70)
print()
print("This demo shows what's REALLY happening inside a tokenizer.")
print("Watch as it builds vocabulary, counts frequencies, and encodes text!")
print()

# =============================================================================
# DEMO 1: Character Tokenizer
# =============================================================================
print("\n" + "🅰️  " + "="*66)
print("DEMO 1: CHARACTER TOKENIZER")
print("="*70)
print("\nThis is the simplest strategy: every unique character gets an ID.")
input("\n👉 Press Enter to see the character tokenizer in action...")

char_tok = SimpleTokenizer(strategy='char', debug=True)
char_tok.train(CORPUS)

print("\n\nNow let's encode a test sentence:")
test_text = "The fox jumps!"
tokens = char_tok.encode(test_text)

print(f"\n📊 Statistics:")
stats = char_tok.get_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

# =============================================================================
# DEMO 2: Word Tokenizer
# =============================================================================
print("\n\n" + "📝 " + "="*66)
print("DEMO 2: WORD TOKENIZER")
print("="*70)
print("\nMore intuitive: each word becomes a token.")
print("Notice how it handles frequency and unknown words!")
input("\n👉 Press Enter to see the word tokenizer in action...")

word_tok = SimpleTokenizer(strategy='word', debug=True)
word_tok.train(CORPUS, vocab_size=50)

print("\n\nEncoding the same sentence:")
tokens = word_tok.encode(test_text)

print(f"\n📊 Statistics:")
stats = word_tok.get_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

# =============================================================================
# DEMO 3: Comparison
# =============================================================================
print("\n\n" + "🆚 " + "="*66)
print("COMPARISON: ALL THREE STRATEGIES")
print("="*70)

# Re-train without debug for clean output
char_tok_clean = SimpleTokenizer(strategy='char')
char_tok_clean.train(CORPUS)

word_tok_clean = SimpleTokenizer(strategy='word')
word_tok_clean.train(CORPUS, vocab_size=100)

bpe_tok_clean = SimpleTokenizer(strategy='bpe')
bpe_tok_clean.train(CORPUS, vocab_size=100)

test_sentences = [
    "The quick fox",
    "Python is fun",
    "Tokenization works"
]

print("\nEncoding different sentences:\n")
for sentence in test_sentences:
    print(f"📝 Sentence: '{sentence}'")

    char_tokens = char_tok_clean.encode(sentence)
    word_tokens = word_tok_clean.encode(sentence)
    bpe_tokens = bpe_tok_clean.encode(sentence)

    print(f"   Character: {len(char_tokens)} tokens → {char_tokens}")
    print(f"   Word:      {len(word_tokens)} tokens → {word_tokens}")
    print(f"   BPE:       {len(bpe_tokens)} tokens → {bpe_tokens}")
    print()

# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "💡 " + "="*66)
print("KEY INSIGHTS FROM UNDER THE HOOD")
print("="*70)
print()
print("1️⃣  CHARACTER tokenizer:")
print("   ✅ Simple, always works")
print("   ⚠️  Long sequences (1 char = 1 token)")
print("   📊 Vocab size: ~100 characters")
print()
print("2️⃣  WORD tokenizer:")
print("   ✅ Intuitive (words = tokens)")
print("   ⚠️  Can't handle unknown words well")
print("   📊 Vocab size: 10,000-100,000 words")
print()
print("3️⃣  BPE tokenizer:")
print("   ✅ Best of both worlds")
print("   ✅ Handles rare words (breaks into subwords)")
print("   📊 Vocab size: 30,000-50,000 subwords")
print()
print("🎓 This is why GPT, BERT, and modern LLMs use BPE!")
print("="*70)

# =============================================================================
# CHALLENGE
# =============================================================================
print("\n" + "🎯 " + "="*66)
print("TRY IT YOURSELF!")
print("="*70)
print()
print("Challenge: What happens with different vocabulary sizes?")
print()
print("Try running:")
print("  tokenizer = SimpleTokenizer(strategy='word', debug=True)")
print("  tokenizer.train(corpus, vocab_size=10)   # Very small")
print("  tokenizer.train(corpus, vocab_size=1000) # Very large")
print()
print("Notice the coverage percentage - what does it mean for your application?")
print()
print("="*70)
