"""
🧪 Workshop 5: Transformer Test Suite
======================================

Tests for the complete transformer implementation.

Usage:
    python test_transformer.py
"""

import numpy as np
from transformer import (
    softmax, layer_norm, gelu,
    PositionalEncoding, MultiHeadAttention, FeedForward,
    TransformerBlock, MiniTransformer, SimpleVocab
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def add(self, name: str, passed: bool, message: str = ""):
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {message}")
        self.results.append((name, passed, message))


def run_tests():
    """Run all transformer tests."""
    results = TestResults()
    
    print("=" * 60)
    print("🧪 TRANSFORMER TEST SUITE")
    print("=" * 60)
    
    # =========================================================================
    # GROUP 1: Utility Functions
    # =========================================================================
    print("\n📦 GROUP 1: Utility Functions")
    print("-" * 40)
    
    # Test softmax
    try:
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert s.shape == x.shape
        assert np.abs(s.sum() - 1.0) < 1e-6
        assert s[2] > s[1] > s[0]  # Preserves order
        results.add("softmax basic", True)
    except Exception as e:
        results.add("softmax basic", False, str(e))
    
    # Test softmax numerical stability
    try:
        x = np.array([1000.0, 1001.0, 1002.0])
        s = softmax(x)
        assert not np.any(np.isnan(s))
        assert not np.any(np.isinf(s))
        results.add("softmax stability", True)
    except Exception as e:
        results.add("softmax stability", False, str(e))
    
    # Test layer_norm
    try:
        x = np.random.randn(5, 64).astype(np.float32)
        normed = layer_norm(x)
        assert normed.shape == x.shape
        # Each row should have mean≈0, var≈1
        row_means = np.abs(normed.mean(axis=-1))
        assert np.all(row_means < 0.01)
        results.add("layer_norm", True)
    except Exception as e:
        results.add("layer_norm", False, str(e))
    
    # Test gelu
    try:
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        g = gelu(x)
        assert g.shape == x.shape
        assert g[2] == 0.0  # gelu(0) = 0
        assert g[4] > g[3] > 0  # Positive values pass through
        assert g[0] < 0 and g[0] > -0.2  # Negative values dampened
        results.add("gelu activation", True)
    except Exception as e:
        results.add("gelu activation", False, str(e))
    
    # =========================================================================
    # GROUP 2: Positional Encoding
    # =========================================================================
    print("\n📦 GROUP 2: Positional Encoding")
    print("-" * 40)
    
    # Test creation
    try:
        pos_enc = PositionalEncoding(embed_dim=64, max_seq_len=128)
        enc = pos_enc(10)
        assert enc.shape == (10, 64)
        results.add("pos_encoding shape", True)
    except Exception as e:
        results.add("pos_encoding shape", False, str(e))
    
    # Test different positions are different
    try:
        pos_enc = PositionalEncoding(embed_dim=64, max_seq_len=128)
        enc = pos_enc(10)
        # Each position should be unique
        for i in range(10):
            for j in range(i + 1, 10):
                diff = np.abs(enc[i] - enc[j]).sum()
                assert diff > 0.1, f"Positions {i} and {j} too similar"
        results.add("pos_encoding uniqueness", True)
    except Exception as e:
        results.add("pos_encoding uniqueness", False, str(e))
    
    # Test sinusoidal pattern
    try:
        pos_enc = PositionalEncoding(embed_dim=64, max_seq_len=128)
        enc = pos_enc(100)
        # Values should be bounded [-1, 1]
        assert np.all(enc >= -1.0) and np.all(enc <= 1.0)
        results.add("pos_encoding bounded", True)
    except Exception as e:
        results.add("pos_encoding bounded", False, str(e))
    
    # =========================================================================
    # GROUP 3: Multi-Head Attention
    # =========================================================================
    print("\n📦 GROUP 3: Multi-Head Attention")
    print("-" * 40)
    
    # Test basic forward
    try:
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(5, 64).astype(np.float32)
        out, weights = attn(x, causal=False)
        assert out.shape == (5, 64)
        assert weights.shape == (4, 5, 5)
        results.add("attention forward", True)
    except Exception as e:
        results.add("attention forward", False, str(e))
    
    # Test causal masking
    try:
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(5, 64).astype(np.float32)
        out, weights = attn(x, causal=True)
        # Upper triangle should be ~0 (masked)
        for h in range(4):
            upper = np.triu(weights[h], k=1)
            assert np.all(upper < 1e-6), "Causal mask not applied"
        results.add("attention causal mask", True)
    except Exception as e:
        results.add("attention causal mask", False, str(e))
    
    # Test attention weights sum to 1
    try:
        attn = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(5, 64).astype(np.float32)
        out, weights = attn(x, causal=False)
        row_sums = weights.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)
        results.add("attention weights sum to 1", True)
    except Exception as e:
        results.add("attention weights sum to 1", False, str(e))
    
    # =========================================================================
    # GROUP 4: Feed-Forward Network
    # =========================================================================
    print("\n📦 GROUP 4: Feed-Forward Network")
    print("-" * 40)
    
    # Test forward
    try:
        ffn = FeedForward(embed_dim=64)
        x = np.random.randn(5, 64).astype(np.float32)
        out = ffn(x)
        assert out.shape == (5, 64)
        results.add("ffn forward", True)
    except Exception as e:
        results.add("ffn forward", False, str(e))
    
    # Test hidden expansion
    try:
        ffn = FeedForward(embed_dim=64)
        assert ffn.hidden_dim == 256  # 4x expansion
        results.add("ffn 4x expansion", True)
    except Exception as e:
        results.add("ffn 4x expansion", False, str(e))
    
    # Test custom hidden dim
    try:
        ffn = FeedForward(embed_dim=64, hidden_dim=128)
        assert ffn.hidden_dim == 128
        results.add("ffn custom hidden", True)
    except Exception as e:
        results.add("ffn custom hidden", False, str(e))
    
    # =========================================================================
    # GROUP 5: Transformer Block
    # =========================================================================
    print("\n📦 GROUP 5: Transformer Block")
    print("-" * 40)
    
    # Test forward
    try:
        block = TransformerBlock(embed_dim=64, num_heads=4)
        x = np.random.randn(5, 64).astype(np.float32)
        out, weights = block(x, causal=True)
        assert out.shape == (5, 64)
        assert weights.shape == (4, 5, 5)
        results.add("block forward", True)
    except Exception as e:
        results.add("block forward", False, str(e))
    
    # Test residual connection (output shouldn't be too different from input)
    try:
        block = TransformerBlock(embed_dim=64, num_heads=4)
        x = np.random.randn(5, 64).astype(np.float32)
        out, _ = block(x, causal=True)
        # With residuals, output should correlate with input
        correlation = np.corrcoef(x.flatten(), out.flatten())[0, 1]
        assert correlation > 0.3, f"Residual connection weak: {correlation}"
        results.add("block residual connection", True)
    except Exception as e:
        results.add("block residual connection", False, str(e))
    
    # =========================================================================
    # GROUP 6: Complete Transformer
    # =========================================================================
    print("\n📦 GROUP 6: Complete Transformer")
    print("-" * 40)
    
    # Test initialization
    try:
        model = MiniTransformer(
            vocab_size=50,
            embed_dim=64,
            num_heads=4,
            num_layers=2
        )
        assert model.vocab_size == 50
        assert model.embed_dim == 64
        assert len(model.blocks) == 2
        results.add("transformer init", True)
    except Exception as e:
        results.add("transformer init", False, str(e))
    
    # Test forward pass
    try:
        model = MiniTransformer(vocab_size=50, embed_dim=64)
        token_ids = [1, 2, 3, 4, 5]
        logits, all_weights = model.forward(token_ids)
        assert logits.shape == (5, 50)  # (seq_len, vocab_size)
        assert len(all_weights) == model.num_layers
        results.add("transformer forward", True)
    except Exception as e:
        results.add("transformer forward", False, str(e))
    
    # Test get_next_token_probs
    try:
        model = MiniTransformer(vocab_size=50, embed_dim=64)
        token_ids = [1, 2, 3]
        probs = model.get_next_token_probs(token_ids)
        assert probs.shape == (50,)
        assert np.abs(probs.sum() - 1.0) < 1e-5
        results.add("transformer next token probs", True)
    except Exception as e:
        results.add("transformer next token probs", False, str(e))
    
    # Test temperature
    try:
        model = MiniTransformer(vocab_size=50, embed_dim=64)
        token_ids = [1, 2, 3]
        probs_low = model.get_next_token_probs(token_ids, temperature=0.5)
        probs_high = model.get_next_token_probs(token_ids, temperature=2.0)
        # Low temp should be more peaked (higher max)
        assert probs_low.max() > probs_high.max()
        results.add("transformer temperature effect", True)
    except Exception as e:
        results.add("transformer temperature effect", False, str(e))
    
    # Test generation
    try:
        model = MiniTransformer(vocab_size=50, embed_dim=64)
        prompt = [1, 2, 3]
        generated = model.generate(prompt, max_new_tokens=5)
        assert len(generated) == len(prompt) + 5
        assert all(0 <= t < 50 for t in generated)
        results.add("transformer generation", True)
    except Exception as e:
        results.add("transformer generation", False, str(e))
    
    # =========================================================================
    # GROUP 7: SimpleVocab
    # =========================================================================
    print("\n📦 GROUP 7: SimpleVocab")
    print("-" * 40)
    
    # Test encode
    try:
        vocab = SimpleVocab()
        ids = vocab.encode("the cat sat")
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)
        results.add("vocab encode", True)
    except Exception as e:
        results.add("vocab encode", False, str(e))
    
    # Test decode
    try:
        vocab = SimpleVocab()
        ids = vocab.encode("the cat sat")
        text = vocab.decode(ids)
        assert text == "the cat sat"
        results.add("vocab decode", True)
    except Exception as e:
        results.add("vocab decode", False, str(e))
    
    # Test unknown words
    try:
        vocab = SimpleVocab()
        ids = vocab.encode("the xyzzy cat")
        assert ids[1] == 1  # <UNK> = 1
        results.add("vocab unknown words", True)
    except Exception as e:
        results.add("vocab unknown words", False, str(e))
    
    # =========================================================================
    # GROUP 8: Integration Tests
    # =========================================================================
    print("\n📦 GROUP 8: Integration Tests")
    print("-" * 40)
    
    # Test full pipeline
    try:
        vocab = SimpleVocab()
        model = MiniTransformer(vocab_size=vocab.vocab_size, embed_dim=64)
        
        prompt = "the cat sat on"
        token_ids = vocab.encode(prompt)
        generated_ids = model.generate(token_ids, max_new_tokens=3)
        output = vocab.decode(generated_ids)
        
        assert len(output.split()) == len(prompt.split()) + 3
        results.add("full pipeline", True)
    except Exception as e:
        results.add("full pipeline", False, str(e))
    
    # Test determinism with seed
    try:
        np.random.seed(123)
        vocab = SimpleVocab()
        model1 = MiniTransformer(vocab_size=vocab.vocab_size, embed_dim=64)
        token_ids = vocab.encode("the dog")
        
        np.random.seed(456)
        gen1 = model1.generate(token_ids.copy(), max_new_tokens=3, temperature=1.0)
        
        np.random.seed(456)
        gen2 = model1.generate(token_ids.copy(), max_new_tokens=3, temperature=1.0)
        
        assert gen1 == gen2, "Same seed should give same output"
        results.add("generation determinism", True)
    except Exception as e:
        results.add("generation determinism", False, str(e))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    
    if results.failed == 0:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Review the output above.")
    
    return results


if __name__ == "__main__":
    run_tests()
