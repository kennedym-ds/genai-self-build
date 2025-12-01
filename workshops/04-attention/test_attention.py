"""
🧪 Workshop 4: Attention Mechanism - Test Suite
================================================

Comprehensive tests for the attention implementation.

Usage:
    python test_attention.py
"""

import numpy as np
import sys


class TestResult:
    """Simple test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []


def run_tests():
    """Run all attention mechanism tests."""
    from attention import SimpleAttention, softmax
    
    results = TestResult()
    
    print("=" * 60)
    print("🧪 ATTENTION MECHANISM TEST SUITE")
    print("=" * 60)
    
    # =========================================================================
    # GROUP 1: Softmax Function Tests
    # =========================================================================
    
    print("\n📦 GROUP 1: Softmax Function")
    print("-" * 40)
    
    # Test 1.1: Basic softmax
    try:
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x)
        assert result.shape == x.shape, f"Shape mismatch: {result.shape}"
        assert np.allclose(result.sum(axis=-1), 1.0), "Doesn't sum to 1"
        assert np.all(result >= 0), "Negative probabilities"
        results.passed += 1
        print("  ✅ Softmax produces valid probabilities")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Softmax basic: {e}")
    
    # Test 1.2: Softmax numerical stability
    try:
        x_large = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax(x_large)
        assert not np.any(np.isnan(result)), "NaN in result"
        assert not np.any(np.isinf(result)), "Inf in result"
        assert np.allclose(result.sum(axis=-1), 1.0), "Doesn't sum to 1"
        results.passed += 1
        print("  ✅ Softmax handles large values (numerically stable)")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Softmax stability: {e}")
    
    # Test 1.3: Softmax with 2D array
    try:
        x_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = softmax(x_2d)
        assert result.shape == x_2d.shape
        assert np.allclose(result.sum(axis=-1), np.ones(3))
        results.passed += 1
        print("  ✅ Softmax works on 2D arrays (row-wise)")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Softmax 2D: {e}")
    
    # =========================================================================
    # GROUP 2: Attention Initialization
    # =========================================================================
    
    print("\n📦 GROUP 2: Initialization")
    print("-" * 40)
    
    # Test 2.1: Basic initialization
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=4)
        assert attn.embed_dim == 64
        assert attn.num_heads == 4
        assert attn.head_dim == 16
        results.passed += 1
        print("  ✅ Basic initialization correct")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Basic init: {e}")
    
    # Test 2.2: Weight matrix shapes
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=4)
        assert attn.W_q.shape == (64, 64), f"W_q shape: {attn.W_q.shape}"
        assert attn.W_k.shape == (64, 64), f"W_k shape: {attn.W_k.shape}"
        assert attn.W_v.shape == (64, 64), f"W_v shape: {attn.W_v.shape}"
        assert attn.W_o.shape == (64, 64), f"W_o shape: {attn.W_o.shape}"
        results.passed += 1
        print("  ✅ Weight matrices have correct shapes")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Weight shapes: {e}")
    
    # Test 2.3: Different configurations
    try:
        configs = [(32, 1), (64, 8), (128, 4), (256, 16)]
        for embed_dim, num_heads in configs:
            attn = SimpleAttention(embed_dim=embed_dim, num_heads=num_heads)
            assert attn.head_dim == embed_dim // num_heads
        results.passed += 1
        print("  ✅ Various embed_dim/num_heads configurations work")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Config variations: {e}")
    
    # =========================================================================
    # GROUP 3: Dot-Product Attention
    # =========================================================================
    
    print("\n📦 GROUP 3: Dot-Product Attention")
    print("-" * 40)
    
    # Test 3.1: Output shape
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=1)
        Q = np.random.randn(10, 64).astype(np.float32)
        K = np.random.randn(10, 64).astype(np.float32)
        V = np.random.randn(10, 64).astype(np.float32)
        
        output, weights = attn.dot_product_attention(Q, K, V)
        assert output.shape == (10, 64), f"Output shape: {output.shape}"
        assert weights.shape == (10, 10), f"Weights shape: {weights.shape}"
        results.passed += 1
        print(f"  ✅ Output shape: {output.shape}, Weights shape: {weights.shape}")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Dot-product shapes: {e}")
    
    # Test 3.2: Weights sum to 1
    try:
        attn = SimpleAttention(embed_dim=32, num_heads=1)
        Q = np.random.randn(5, 32).astype(np.float32)
        K = np.random.randn(5, 32).astype(np.float32)
        V = np.random.randn(5, 32).astype(np.float32)
        
        _, weights = attn.dot_product_attention(Q, K, V)
        row_sums = weights.sum(axis=-1)
        assert np.allclose(row_sums, 1.0), f"Row sums: {row_sums}"
        results.passed += 1
        print(f"  ✅ All attention weights sum to 1.0")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Weights sum: {e}")
    
    # Test 3.3: Different sequence lengths for Q and K
    try:
        attn = SimpleAttention(embed_dim=32, num_heads=1)
        Q = np.random.randn(5, 32).astype(np.float32)
        K = np.random.randn(8, 32).astype(np.float32)
        V = np.random.randn(8, 32).astype(np.float32)
        
        output, weights = attn.dot_product_attention(Q, K, V)
        assert output.shape == (5, 32), f"Output: {output.shape}"
        assert weights.shape == (5, 8), f"Weights: {weights.shape}"
        results.passed += 1
        print(f"  ✅ Cross-attention shapes correct (Q:5, K:8)")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Cross-attention: {e}")
    
    # =========================================================================
    # GROUP 4: Scaled Dot-Product Attention
    # =========================================================================
    
    print("\n📦 GROUP 4: Scaled Dot-Product Attention")
    print("-" * 40)
    
    # Test 4.1: Scaling effect
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=1)
        Q = np.random.randn(5, 64).astype(np.float32) * 10  # Large values
        K = Q.copy()
        V = np.random.randn(5, 64).astype(np.float32)
        
        _, weights_unscaled = attn.dot_product_attention(Q, K, V)
        _, weights_scaled = attn.scaled_dot_product_attention(Q, K, V)
        
        # Scaled should have higher entropy (more distributed)
        entropy_unscaled = -np.sum(weights_unscaled * np.log(weights_unscaled + 1e-10)) / len(Q)
        entropy_scaled = -np.sum(weights_scaled * np.log(weights_scaled + 1e-10)) / len(Q)
        
        results.passed += 1
        print(f"  ✅ Scaling works - entropy: unscaled={entropy_unscaled:.2f}, scaled={entropy_scaled:.2f}")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Scaling effect: {e}")
    
    # Test 4.2: Masking
    try:
        attn = SimpleAttention(embed_dim=32, num_heads=1)
        Q = np.random.randn(4, 32).astype(np.float32)
        K = np.random.randn(4, 32).astype(np.float32)
        V = np.random.randn(4, 32).astype(np.float32)
        
        # Create a mask that blocks last 2 positions
        mask = np.array([
            [0, 0, -1e9, -1e9],
            [0, 0, -1e9, -1e9],
            [0, 0, -1e9, -1e9],
            [0, 0, -1e9, -1e9],
        ])
        
        _, weights = attn.scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # Weights in masked positions should be near zero
        assert np.all(weights[:, 2:] < 0.01), f"Masked positions not near zero: {weights[:, 2:]}"
        results.passed += 1
        print("  ✅ Masking correctly zeros out blocked positions")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Masking: {e}")
    
    # =========================================================================
    # GROUP 5: Causal Masking
    # =========================================================================
    
    print("\n📦 GROUP 5: Causal Masking")
    print("-" * 40)
    
    # Test 5.1: Causal mask shape
    try:
        mask = SimpleAttention.create_causal_mask(5)
        assert mask.shape == (5, 5), f"Mask shape: {mask.shape}"
        results.passed += 1
        print(f"  ✅ Causal mask shape: {mask.shape}")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Causal mask shape: {e}")
    
    # Test 5.2: Causal mask values
    try:
        mask = SimpleAttention.create_causal_mask(4)
        
        # Upper triangle should be -inf (or very negative)
        for i in range(4):
            for j in range(4):
                if j > i:
                    assert mask[i, j] < -1e8, f"Position ({i},{j}) not masked"
                else:
                    assert mask[i, j] == 0, f"Position ({i},{j}) incorrectly masked"
        
        results.passed += 1
        print("  ✅ Causal mask correctly blocks future positions")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Causal mask values: {e}")
    
    # Test 5.3: Causal attention output
    try:
        attn = SimpleAttention(embed_dim=32, num_heads=1)
        x = np.random.randn(6, 32).astype(np.float32)
        
        output, weights = attn.forward(x, causal=True)
        
        # Check that future positions have zero attention
        for i in range(6):
            for j in range(i + 1, 6):
                assert weights[i, j] < 0.001, f"Future attention at ({i},{j}): {weights[i, j]}"
        
        results.passed += 1
        print("  ✅ Causal attention blocks future correctly")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Causal attention: {e}")
    
    # =========================================================================
    # GROUP 6: Multi-Head Attention
    # =========================================================================
    
    print("\n📦 GROUP 6: Multi-Head Attention")
    print("-" * 40)
    
    # Test 6.1: Multi-head output shape
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(8, 64).astype(np.float32)
        
        output, weights = attn.multi_head_attention(x, x, x)
        
        assert output.shape == (8, 64), f"Output shape: {output.shape}"
        assert weights.shape == (4, 8, 8), f"Weights shape: {weights.shape}"
        results.passed += 1
        print(f"  ✅ Multi-head output: {output.shape}, weights: {weights.shape}")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Multi-head shapes: {e}")
    
    # Test 6.2: Each head has valid weights
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=8)
        x = np.random.randn(5, 64).astype(np.float32)
        
        _, weights = attn.multi_head_attention(x, x, x)
        
        for h in range(8):
            head_weights = weights[h]
            row_sums = head_weights.sum(axis=-1)
            assert np.allclose(row_sums, 1.0), f"Head {h} sums: {row_sums}"
        
        results.passed += 1
        print("  ✅ All 8 heads produce valid probability distributions")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Head weight validity: {e}")
    
    # Test 6.3: Heads are different
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(6, 64).astype(np.float32)
        
        _, weights = attn.multi_head_attention(x, x, x)
        
        # Check that heads are not all identical
        differences = 0
        for h1 in range(4):
            for h2 in range(h1 + 1, 4):
                if not np.allclose(weights[h1], weights[h2], atol=0.1):
                    differences += 1
        
        assert differences > 0, "All heads are identical!"
        results.passed += 1
        print(f"  ✅ Heads produce different attention patterns ({differences} pairs differ)")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Head diversity: {e}")
    
    # =========================================================================
    # GROUP 7: Self-Attention
    # =========================================================================
    
    print("\n📦 GROUP 7: Self-Attention")
    print("-" * 40)
    
    # Test 7.1: Self-attention basic
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(10, 64).astype(np.float32)
        
        output, weights = attn.self_attention(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        results.passed += 1
        print(f"  ✅ Self-attention preserves shape: {output.shape}")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Self-attention basic: {e}")
    
    # Test 7.2: Self-attention with different sizes
    try:
        attn = SimpleAttention(embed_dim=128, num_heads=8)
        
        for seq_len in [1, 5, 10, 50]:
            x = np.random.randn(seq_len, 128).astype(np.float32)
            output, _ = attn.self_attention(x)
            assert output.shape == x.shape
        
        results.passed += 1
        print("  ✅ Self-attention works with various sequence lengths (1, 5, 10, 50)")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Self-attention sizes: {e}")
    
    # =========================================================================
    # GROUP 8: Forward Pass
    # =========================================================================
    
    print("\n📦 GROUP 8: Forward Pass")
    print("-" * 40)
    
    # Test 8.1: Forward with default settings
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(8, 64).astype(np.float32)
        
        output, weights = attn.forward(x)
        
        assert output.shape == x.shape
        results.passed += 1
        print(f"  ✅ Forward pass works: {output.shape}")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Forward basic: {e}")
    
    # Test 8.2: Forward with causal
    try:
        attn = SimpleAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(8, 64).astype(np.float32)
        
        output_bi, _ = attn.forward(x, causal=False)
        output_causal, _ = attn.forward(x, causal=True)
        
        # They should be different
        assert not np.allclose(output_bi, output_causal)
        results.passed += 1
        print("  ✅ Causal vs bidirectional produce different outputs")
    except Exception as e:
        results.failed += 1
        print(f"  ❌ Forward causal: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 60)
    total = results.passed + results.failed
    
    if results.failed == 0:
        print(f"🎉 ALL TESTS PASSED! ({results.passed}/{total})")
    else:
        print(f"📊 RESULTS: {results.passed} passed, {results.failed} failed")
    
    print("=" * 60)
    
    return results.failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
