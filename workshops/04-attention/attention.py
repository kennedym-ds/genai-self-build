"""
🎯 Workshop 4: Attention Mechanism
==================================

👀 THE SPOTLIGHT ANALOGY:
Imagine our alien is at a concert with thousands of musicians playing at once.
How do they focus on the important sounds?

They have a MAGICAL SPOTLIGHT that can:
1. Shine on multiple musicians at different intensities
2. Adjust brightness based on what they're trying to understand
3. Combine sounds from spotlit musicians into one clear signal

When hearing "The cat sat on the mat because it was tired":
- To understand "it" → spotlight shines BRIGHT on "cat" (not "mat")
- The alien learns: "it" ATTENDS to "cat"

THIS IS EXACTLY WHAT ATTENTION DOES.
Each word looks at ALL other words and decides which ones matter most.

THREE TYPES WE'LL EXPLORE:
1. Dot-Product Attention: Simple similarity-based focus
2. Scaled Dot-Product: Prevents extreme values in high dimensions
3. Multi-Head Attention: Multiple spotlights looking for different patterns

Usage:
    python attention.py
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax values - converts scores to probabilities.
    
    Why softmax?
    - Turns any numbers into probabilities (0 to 1, sum to 1)
    - Larger values get exponentially more weight
    - Perfect for "how much should I attend to each word?"
    
    Example:
        scores = [2.0, 1.0, 0.1]
        softmax → [0.659, 0.242, 0.099]  # Highest score gets most attention
    """
    # Subtract max for numerical stability (prevents overflow)
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class SimpleAttention:
    """
    A from-scratch implementation of attention mechanisms.
    
    The core idea: for each position, compute a weighted sum of all positions,
    where weights are based on similarity (relevance).
    
    Example usage:
        attention = SimpleAttention(embed_dim=64)
        
        # Input: sequence of word embeddings [seq_len, embed_dim]
        embeddings = np.random.randn(10, 64)
        
        # Output: contextualized embeddings (same shape)
        output, weights = attention.forward(embeddings)
    """
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 1):
        """
        Initialize attention mechanism.
        
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads (for multi-head attention)
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        # Initialize projection matrices (Query, Key, Value)
        # In real transformers, these are learned parameters
        np.random.seed(42)
        scale = np.sqrt(2.0 / embed_dim)
        
        self.W_q = np.random.randn(embed_dim, embed_dim) * scale  # Query projection
        self.W_k = np.random.randn(embed_dim, embed_dim) * scale  # Key projection
        self.W_v = np.random.randn(embed_dim, embed_dim) * scale  # Value projection
        self.W_o = np.random.randn(embed_dim, embed_dim) * scale  # Output projection
    
    # =========================================================================
    # PART 1: DOT-PRODUCT ATTENTION (Core concept!)
    # =========================================================================
    
    def dot_product_attention(
        self, 
        query: np.ndarray, 
        key: np.ndarray, 
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basic dot-product attention.
        
        The formula: Attention(Q, K, V) = softmax(Q @ K^T) @ V
        
        Step by step:
        1. Compute similarity scores: Q @ K^T (how similar is each query to each key?)
        2. Convert to probabilities: softmax (normalize to sum to 1)
        3. Weighted sum of values: scores @ V (combine values based on attention)
        
        Args:
            query: What we're looking for [seq_len, dim]
            key: What we're comparing against [seq_len, dim]
            value: What we'll retrieve [seq_len, dim]
            mask: Optional mask to hide certain positions
            
        Returns:
            output: Attended values [seq_len, dim]
            attention_weights: Who attended to whom [seq_len, seq_len]
        
        Example:
            For "The cat sat on the mat":
            - Query for "sat": What context do I need?
            - Keys for all words: What context can each word provide?
            - Similarity: "sat" is most similar to "cat" (subject) and "mat" (location)
            - Values: Blend information from attended words
        """
        # Step 1: Compute attention scores (similarity)
        # [seq_len, dim] @ [dim, seq_len] = [seq_len, seq_len]
        scores = np.matmul(query, key.T)
        
        # Step 2: Apply mask if provided (e.g., for causal attention)
        if mask is not None:
            scores = scores + mask  # Mask contains -inf where we shouldn't attend
        
        # Step 3: Convert to probabilities
        attention_weights = softmax(scores, axis=-1)
        
        # Step 4: Weighted sum of values
        # [seq_len, seq_len] @ [seq_len, dim] = [seq_len, dim]
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
    
    # =========================================================================
    # PART 2: SCALED DOT-PRODUCT ATTENTION (What transformers use)
    # =========================================================================
    
    def scaled_dot_product_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention (from "Attention Is All You Need").
        
        Why scale? In high dimensions, dot products can get very large,
        making softmax produce extreme values (almost all 0s and one 1).
        
        Scaling by sqrt(d_k) keeps gradients healthy during training.
        
        Formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        Args:
            query: [seq_len, dim] or [batch, seq_len, dim]
            key: [seq_len, dim] or [batch, seq_len, dim]  
            value: [seq_len, dim] or [batch, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            output: Attended values
            attention_weights: Attention distribution
        """
        # Get dimension for scaling
        d_k = query.shape[-1]
        
        # Compute scaled scores
        scores = np.matmul(query, key.swapaxes(-2, -1)) / np.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores + mask
        
        # Softmax to get attention weights
        attention_weights = softmax(scores, axis=-1)
        
        # Weighted sum of values
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
    
    # =========================================================================
    # PART 3: MULTI-HEAD ATTENTION (Multiple perspectives)
    # =========================================================================
    
    def multi_head_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-head attention - multiple attention "heads" looking for different patterns.
        
        Why multiple heads?
        - One head might focus on syntax (subject-verb relationships)
        - Another might focus on semantics (word meanings)
        - Another might focus on position (nearby words)
        
        Each head has its own Q, K, V projections and sees the data differently.
        
        Args:
            query: Input for queries [seq_len, embed_dim]
            key: Input for keys [seq_len, embed_dim]
            value: Input for values [seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            output: Multi-head attended values [seq_len, embed_dim]
            attention_weights: Weights from all heads [num_heads, seq_len, seq_len]
        """
        seq_len = query.shape[0]
        
        # Step 1: Project to queries, keys, values
        Q = np.matmul(query, self.W_q)  # [seq_len, embed_dim]
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # Step 2: Reshape into multiple heads
        # [seq_len, embed_dim] -> [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        
        # Step 3: Apply scaled attention for each head
        # Q, K, V are now [num_heads, seq_len, head_dim]
        head_outputs = []
        all_weights = []
        
        for h in range(self.num_heads):
            head_out, weights = self.scaled_dot_product_attention(
                Q[h], K[h], V[h], mask
            )
            head_outputs.append(head_out)
            all_weights.append(weights)
        
        # Step 4: Concatenate heads
        # [num_heads, seq_len, head_dim] -> [seq_len, embed_dim]
        concat = np.concatenate(head_outputs, axis=-1)
        
        # Step 5: Final projection
        output = np.matmul(concat, self.W_o)
        
        return output, np.array(all_weights)
    
    # =========================================================================
    # PART 4: SELF-ATTENTION (The key to transformers)
    # =========================================================================
    
    def self_attention(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_multihead: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Self-attention: each position attends to ALL positions (including itself).
        
        This is what makes transformers so powerful:
        - Every word can directly attend to every other word
        - No sequential bottleneck like RNNs
        - Captures long-range dependencies naturally
        
        Args:
            x: Input embeddings [seq_len, embed_dim]
            mask: Optional attention mask
            use_multihead: Whether to use multi-head attention
            
        Returns:
            output: Contextualized embeddings [seq_len, embed_dim]
            attention_weights: Attention pattern
        """
        if use_multihead:
            return self.multi_head_attention(x, x, x, mask)
        else:
            return self.scaled_dot_product_attention(x, x, x, mask)
    
    # =========================================================================
    # UTILITY: Causal Mask (for autoregressive models like GPT)
    # =========================================================================
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """
        Create a causal (look-ahead) mask for autoregressive attention.
        
        In language models like GPT, when predicting the next word,
        we can only look at PREVIOUS words, not future ones.
        
        Example for seq_len=4:
            [[0, -inf, -inf, -inf],   # Position 0 can only see position 0
             [0,    0, -inf, -inf],   # Position 1 can see 0, 1
             [0,    0,    0, -inf],   # Position 2 can see 0, 1, 2
             [0,    0,    0,    0]]   # Position 3 can see all
        
        After softmax, -inf becomes 0 (no attention).
        """
        mask = np.triu(np.ones((seq_len, seq_len)) * float('-inf'), k=1)
        return mask
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        causal: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main forward pass.
        
        Args:
            x: Input embeddings [seq_len, embed_dim]
            mask: Optional custom mask
            causal: If True, use causal masking (for autoregressive models)
            
        Returns:
            output: Contextualized embeddings
            attention_weights: Attention distribution
        """
        if causal:
            seq_len = x.shape[0]
            mask = self.create_causal_mask(seq_len)
        
        return self.self_attention(x, mask, use_multihead=self.num_heads > 1)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def visualize_attention_text(
    words: List[str],
    attention_weights: np.ndarray,
    query_idx: int = 0
) -> str:
    """
    Create a text visualization of attention weights.
    
    Args:
        words: List of words in the sequence
        attention_weights: Attention matrix [seq_len, seq_len]
        query_idx: Which word's attention to show
        
    Returns:
        Formatted string showing attention distribution
    """
    weights = attention_weights[query_idx]
    
    lines = [f"\n👀 Attention from '{words[query_idx]}' to other words:"]
    lines.append("-" * 50)
    
    # Sort by attention weight
    sorted_indices = np.argsort(weights)[::-1]
    
    for idx in sorted_indices:
        weight = weights[idx]
        bar_len = int(weight * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        
        marker = " ← (self)" if idx == query_idx else ""
        lines.append(f"  {words[idx]:15s} [{bar}] {weight:.3f}{marker}")
    
    return "\n".join(lines)


def create_sample_embeddings(words: List[str], embed_dim: int = 64) -> np.ndarray:
    """
    Create simple embeddings for demonstration.
    
    In practice, these would come from Workshop 2 (trained embeddings).
    Here we use random embeddings with some structure.
    """
    np.random.seed(hash(tuple(words)) % 2**32)
    embeddings = np.random.randn(len(words), embed_dim)
    
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    
    return embeddings


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demo_basic_attention():
    """Demonstrate basic dot-product attention."""
    print("\n" + "=" * 60)
    print("🎯 DEMO 1: Basic Dot-Product Attention")
    print("=" * 60)
    
    words = ["The", "cat", "sat", "on", "the", "mat"]
    embed_dim = 32
    
    # Create embeddings
    embeddings = create_sample_embeddings(words, embed_dim)
    
    print(f"\n📝 Input: '{' '.join(words)}'")
    print(f"📐 Embedding dimension: {embed_dim}")
    print(f"📊 Sequence length: {len(words)}")
    
    # Apply attention
    attention = SimpleAttention(embed_dim=embed_dim, num_heads=1)
    output, weights = attention.scaled_dot_product_attention(
        embeddings, embeddings, embeddings
    )
    
    print(f"\n✅ Output shape: {output.shape}")
    print(f"📊 Attention weights shape: {weights.shape}")
    
    # Visualize attention for "sat"
    print(visualize_attention_text(words, weights, query_idx=2))
    
    return weights


def demo_multi_head_attention():
    """Demonstrate multi-head attention."""
    print("\n" + "=" * 60)
    print("🎯 DEMO 2: Multi-Head Attention (4 heads)")
    print("=" * 60)
    
    words = ["The", "cat", "sat", "on", "the", "mat"]
    embed_dim = 64
    num_heads = 4
    
    embeddings = create_sample_embeddings(words, embed_dim)
    
    print(f"\n📝 Input: '{' '.join(words)}'")
    print(f"📐 Embedding dimension: {embed_dim}")
    print(f"🔢 Number of heads: {num_heads}")
    print(f"📊 Head dimension: {embed_dim // num_heads}")
    
    attention = SimpleAttention(embed_dim=embed_dim, num_heads=num_heads)
    output, all_weights = attention.multi_head_attention(
        embeddings, embeddings, embeddings
    )
    
    print(f"\n✅ Output shape: {output.shape}")
    print(f"📊 All attention weights shape: {all_weights.shape}")
    
    # Show each head's attention for "sat"
    query_idx = 2
    print(f"\n👀 How each head attends from '{words[query_idx]}':")
    
    for h in range(num_heads):
        weights = all_weights[h][query_idx]
        top_idx = np.argmax(weights)
        print(f"\n  Head {h + 1}: Focuses most on '{words[top_idx]}' ({weights[top_idx]:.3f})")
        
        # Show top 3
        top_3 = np.argsort(weights)[::-1][:3]
        for idx in top_3:
            bar_len = int(weights[idx] * 20)
            bar = "█" * bar_len
            print(f"    {words[idx]:10s} [{bar:20s}] {weights[idx]:.3f}")


def demo_causal_attention():
    """Demonstrate causal (autoregressive) attention."""
    print("\n" + "=" * 60)
    print("🎯 DEMO 3: Causal Attention (GPT-style)")
    print("=" * 60)
    
    words = ["The", "cat", "sat", "on", "the", "mat"]
    embed_dim = 32
    
    embeddings = create_sample_embeddings(words, embed_dim)
    
    print(f"\n📝 Input: '{' '.join(words)}'")
    print("\n🎭 Causal Mask (positions can only see past, not future):")
    
    mask = SimpleAttention.create_causal_mask(len(words))
    
    # Show mask visually
    print("\n     " + "  ".join(f"{w[:3]:>3s}" for w in words))
    for i, word in enumerate(words):
        row = []
        for j in range(len(words)):
            if mask[i, j] == float('-inf'):
                row.append(" ✗ ")
            else:
                row.append(" ✓ ")
        print(f"{word[:3]:>3s}  " + " ".join(row))
    
    print("\n✓ = can attend, ✗ = masked (can't see future)")
    
    # Apply causal attention
    attention = SimpleAttention(embed_dim=embed_dim, num_heads=1)
    output, weights = attention.forward(embeddings, causal=True)
    
    print(f"\n📊 Attention weights with causal mask:")
    print("   (Each row shows what that position attends to)")
    
    for i, word in enumerate(words):
        row_weights = weights[i]
        nonzero = row_weights[row_weights > 0.01]
        print(f"\n  '{word}' attends to: ", end="")
        for j, w in enumerate(row_weights):
            if w > 0.01:
                print(f"{words[j]}({w:.2f}) ", end="")


def demo_attention_interpretation():
    """Show how attention reveals relationships."""
    print("\n" + "=" * 60)
    print("🎯 DEMO 4: Attention Reveals Relationships")
    print("=" * 60)
    
    # Example with pronouns
    words = ["The", "cat", "sat", "because", "it", "was", "tired"]
    embed_dim = 32
    
    print(f"\n📝 Sentence: '{' '.join(words)}'")
    print("\n❓ Question: What does 'it' refer to?")
    
    # Create embeddings where "it" is similar to "cat"
    np.random.seed(42)
    embeddings = np.random.randn(len(words), embed_dim)
    
    # Make "it" (index 4) similar to "cat" (index 1)
    embeddings[4] = embeddings[1] + np.random.randn(embed_dim) * 0.3
    
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    
    attention = SimpleAttention(embed_dim=embed_dim, num_heads=1)
    output, weights = attention.scaled_dot_product_attention(
        embeddings, embeddings, embeddings
    )
    
    # Show what "it" attends to
    it_idx = 4
    print(visualize_attention_text(words, weights, query_idx=it_idx))
    
    top_idx = np.argmax(weights[it_idx])
    if top_idx != it_idx:
        print(f"\n✅ 'it' attends most strongly to '{words[top_idx]}'!")
        print("   This is how transformers resolve pronoun references.")


# =============================================================================
# TEST SUITE
# =============================================================================

def run_tests():
    """Run all tests for the attention implementation."""
    print("\n" + "=" * 60)
    print("🧪 ATTENTION MECHANISM TEST SUITE")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Softmax
    print("\n📦 TEST 1: Softmax")
    try:
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.allclose(result.sum(), 1.0), "Softmax should sum to 1"
        assert result[2] > result[1] > result[0], "Larger inputs should have larger outputs"
        print("  ✅ Softmax produces valid probabilities")
        passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed += 1
    
    # Test 2: Basic attention shapes
    print("\n📦 TEST 2: Attention Shapes")
    try:
        seq_len, embed_dim = 10, 64
        x = np.random.randn(seq_len, embed_dim)
        
        attention = SimpleAttention(embed_dim=embed_dim)
        output, weights = attention.forward(x)
        
        assert output.shape == (seq_len, embed_dim), f"Output shape mismatch"
        assert weights.shape == (seq_len, seq_len), f"Weights shape mismatch"
        print(f"  ✅ Output shape: {output.shape}")
        print(f"  ✅ Weights shape: {weights.shape}")
        passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed += 1
    
    # Test 3: Attention weights sum to 1
    print("\n📦 TEST 3: Attention Weights Sum to 1")
    try:
        seq_len, embed_dim = 5, 32
        x = np.random.randn(seq_len, embed_dim)
        
        attention = SimpleAttention(embed_dim=embed_dim)
        _, weights = attention.forward(x)
        
        row_sums = weights.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Each row should sum to 1"
        print(f"  ✅ All row sums ≈ 1.0: {row_sums}")
        passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed += 1
    
    # Test 4: Multi-head attention
    print("\n📦 TEST 4: Multi-Head Attention")
    try:
        seq_len, embed_dim, num_heads = 6, 64, 4
        x = np.random.randn(seq_len, embed_dim)
        
        attention = SimpleAttention(embed_dim=embed_dim, num_heads=num_heads)
        output, weights = attention.forward(x)
        
        assert output.shape == (seq_len, embed_dim)
        assert weights.shape == (num_heads, seq_len, seq_len)
        print(f"  ✅ Multi-head output shape: {output.shape}")
        print(f"  ✅ Multi-head weights shape: {weights.shape}")
        passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed += 1
    
    # Test 5: Causal mask
    print("\n📦 TEST 5: Causal Mask")
    try:
        mask = SimpleAttention.create_causal_mask(4)
        
        # Upper triangle should be -inf
        assert mask[0, 1] == float('-inf'), "Position 0 shouldn't see position 1"
        assert mask[0, 0] == 0, "Position 0 should see itself"
        assert mask[3, 0] == 0, "Position 3 should see position 0"
        print("  ✅ Causal mask correctly blocks future positions")
        passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed += 1
    
    # Test 6: Causal attention
    print("\n📦 TEST 6: Causal Attention")
    try:
        seq_len, embed_dim = 5, 32
        x = np.random.randn(seq_len, embed_dim)
        
        attention = SimpleAttention(embed_dim=embed_dim)
        _, weights = attention.forward(x, causal=True)
        
        # Check that future positions have 0 attention
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[i, j] < 1e-6, f"Position {i} shouldn't attend to {j}"
        
        print("  ✅ Causal attention correctly masks future")
        passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed += 1
    
    # Test 7: Self-attention is symmetric input/output
    print("\n📦 TEST 7: Self-Attention Preserves Dimensions")
    try:
        for seq_len in [1, 5, 20]:
            for embed_dim in [16, 64, 128]:
                x = np.random.randn(seq_len, embed_dim)
                attention = SimpleAttention(embed_dim=embed_dim)
                output, _ = attention.forward(x)
                assert output.shape == x.shape
        
        print("  ✅ Various input sizes work correctly")
        passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
    else:
        print(f"⚠️  {passed}/{total} tests passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run tests
    run_tests()
    
    # Run demos
    demo_basic_attention()
    demo_multi_head_attention()
    demo_causal_attention()
    demo_attention_interpretation()
    
    print("\n" + "=" * 60)
    print("🎓 KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. ATTENTION computes weighted averages based on relevance
       - Query: What am I looking for?
       - Key: What can each position offer?
       - Value: What information to retrieve?
    
    2. SCALED attention prevents extreme values in high dimensions
       - Divide by sqrt(d_k) to keep gradients stable
    
    3. MULTI-HEAD attention captures different relationship types
       - Each head can learn different patterns
       - Concat and project back to original dimension
    
    4. CAUSAL attention is for autoregressive models (GPT)
       - Can only attend to past, not future
       - Essential for next-token prediction
    
    5. SELF-ATTENTION is the key to transformers
       - Every position can directly attend to every other
       - No sequential bottleneck like RNNs
       - Captures long-range dependencies naturally
    """)
