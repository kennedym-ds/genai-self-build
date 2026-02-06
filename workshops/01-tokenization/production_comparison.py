#!/usr/bin/env python3
"""
🔍 Production Comparison Demo
==============================

This script compares our educational tokenizer with production systems
to show what's different and why.

Note: This requires tiktoken to be installed:
    pip install tiktoken

If tiktoken is not available, the script will still demonstrate our tokenizer.
"""

from tokenizer import SimpleTokenizer

# Try to import production tokenizer
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("⚠️  tiktoken not installed. Install with: pip install tiktoken")
    print("    (Continuing with demo of our tokenizer only)\n")

# Sample texts for comparison
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! How are you today?",
    "Machine learning and artificial intelligence are transforming technology.",
    "GPT-4 uses byte-pair encoding with a vocabulary of 100,000 tokens.",
    "🚀 Emojis and special characters: 你好世界 مرحبا 123!@#",
]

# Training corpus for our tokenizer
TRAINING_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming how we build software.",
    "Python is a popular programming language.",
    "Tokenization is the first step in natural language processing.",
    "Artificial intelligence and neural networks are powerful tools.",
    "The fox is quick and clever, always learning new tricks.",
]

print("=" * 70)
print("🔍 EDUCATIONAL vs PRODUCTION TOKENIZER COMPARISON")
print("=" * 70)
print()

# =============================================================================
# PART 1: Our Implementation
# =============================================================================
print("📚 PART 1: Our Educational BPE Tokenizer")
print("-" * 70)
print()

our_tokenizer = SimpleTokenizer(strategy='bpe', debug=False)
print("Training our tokenizer on small corpus (6 documents)...")
our_tokenizer.train(TRAINING_CORPUS, vocab_size=200)

stats = our_tokenizer.get_stats()
print(f"\n📊 Our Tokenizer Stats:")
print(f"   Strategy: {stats['strategy']}")
print(f"   Vocabulary size: {stats['vocab_size']}")
print(f"   Training corpus: {len(TRAINING_CORPUS)} documents")
if 'num_merges' in stats:
    print(f"   BPE merges learned: {stats['num_merges']}")
print()

# =============================================================================
# PART 2: Production Implementation (if available)
# =============================================================================
if HAS_TIKTOKEN:
    print("\n🏭 PART 2: Production Tokenizer (GPT-4's tiktoken)")
    print("-" * 70)
    print()

    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    print("📊 GPT-4 Tokenizer Stats:")
    print(f"   Vocabulary size: ~100,000 tokens")
    print(f"   Training corpus: Billions of web documents")
    print(f"   Algorithm: Byte-level BPE")
    print(f"   Implementation: Rust (optimized for speed)")
    print()

# =============================================================================
# PART 3: Side-by-Side Comparison
# =============================================================================
print("\n🆚 PART 3: Side-by-Side Comparison")
print("=" * 70)
print()

for i, text in enumerate(SAMPLE_TEXTS, 1):
    print(f"Example {i}: \"{text}\"")
    print("-" * 70)

    # Our tokenizer
    our_tokens = our_tokenizer.encode(text)
    our_decoded = our_tokenizer.decode(our_tokens)

    print(f"📚 Our BPE:")
    print(f"   Tokens: {len(our_tokens)}")
    print(f"   IDs: {our_tokens[:15]}{'...' if len(our_tokens) > 15 else ''}")
    print(f"   Decoded: \"{our_decoded}\"")

    # Production tokenizer
    if HAS_TIKTOKEN:
        gpt_tokens = gpt_tokenizer.encode(text)
        gpt_decoded = gpt_tokenizer.decode(gpt_tokens)

        print(f"🏭 GPT-4:")
        print(f"   Tokens: {len(gpt_tokens)}")
        print(f"   IDs: {gpt_tokens[:15]}{'...' if len(gpt_tokens) > 15 else ''}")
        print(f"   Decoded: \"{gpt_decoded}\"")

        # Comparison
        efficiency = len(our_tokens) / len(gpt_tokens) if len(gpt_tokens) > 0 else 0
        print(f"\n📊 Comparison:")
        print(f"   Our tokenizer uses {efficiency:.2f}x more tokens")
        if efficiency > 1:
            print(f"   (GPT-4 is more efficient due to larger vocabulary)")

    print()

# =============================================================================
# PART 4: Key Differences Explained
# =============================================================================
print("\n💡 PART 4: Key Differences Explained")
print("=" * 70)
print()

differences = [
    ("Vocabulary Size", "200 tokens", "100,000 tokens",
     "Larger vocab = fewer tokens per text"),

    ("Training Data", "6 documents", "Billions of documents",
     "More data = better coverage of patterns"),

    ("Algorithm", "Character-level BPE", "Byte-level BPE",
     "Bytes handle all Unicode, chars are limited"),

    ("Implementation", "Pure Python", "Rust (compiled)",
     "Rust is 10,000x faster for production use"),

    ("Special Tokens", "Basic (<UNK>, <PAD>)", "Many specialized tokens",
     "Production needs <|endoftext|>, <|im_start|>, etc."),

    ("Pre-tokenization", "Simple word split", "Complex regex patterns",
     "Handles punctuation, whitespace better"),
]

for category, ours, production, explanation in differences:
    print(f"🔹 {category}:")
    print(f"   📚 Our implementation: {ours}")
    if HAS_TIKTOKEN:
        print(f"   🏭 Production: {production}")
    print(f"   💡 Why it matters: {explanation}")
    print()

# =============================================================================
# PART 5: When to Use Which
# =============================================================================
print("\n🎯 PART 5: When to Use Which?")
print("=" * 70)
print()

print("📚 Use Our Educational Tokenizer When:")
print("   ✅ Learning how tokenization works")
print("   ✅ Understanding BPE algorithm step-by-step")
print("   ✅ Building proof-of-concept systems")
print("   ✅ Teaching or demonstrating concepts")
print("   ✅ Working with small, controlled datasets")
print()

if HAS_TIKTOKEN:
    print("🏭 Use Production Tokenizer (tiktoken) When:")
    print("   ✅ Building production applications")
    print("   ✅ Need to be compatible with GPT models")
    print("   ✅ Processing large amounts of text")
    print("   ✅ Working with multilingual content")
    print("   ✅ Performance and efficiency matter")
    print()

# =============================================================================
# PART 6: Try It Yourself
# =============================================================================
print("\n🎓 PART 6: Try It Yourself!")
print("=" * 70)
print()

print("Experiment 1: Vocabulary Size Impact")
print("-------------------------------------")
print("Try training with different vocab sizes:")
print()
print("  small = SimpleTokenizer(strategy='bpe')")
print("  small.train(corpus, vocab_size=50)")
print()
print("  large = SimpleTokenizer(strategy='bpe')")
print("  large.train(corpus, vocab_size=500)")
print()
print("Question: How does vocab size affect token count?")
print()

print("Experiment 2: Debug Mode")
print("------------------------")
print("See exactly what's happening inside:")
print()
print("  debug_tok = SimpleTokenizer(strategy='bpe', debug=True)")
print("  debug_tok.train(corpus, vocab_size=100)")
print()
print("Watch the BPE merge iterations in real-time!")
print()

print("Experiment 3: Compare Your Own Text")
print("------------------------------------")
if HAS_TIKTOKEN:
    print("  your_text = 'Your custom text here...'")
    print("  our_tokens = our_tokenizer.encode(your_text)")
    print("  gpt_tokens = gpt_tokenizer.encode(your_text)")
    print("  print(f'Ours: {len(our_tokens)}, GPT: {len(gpt_tokens)}')")
else:
    print("  your_text = 'Your custom text here...'")
    print("  tokens = our_tokenizer.encode(your_text)")
    print("  print(f'Tokens: {tokens}')")
    print("  print(f'Decoded: {our_tokenizer.decode(tokens)}')")
print()

print("=" * 70)
print("🎉 Now you understand the difference between educational and production!")
print("=" * 70)
