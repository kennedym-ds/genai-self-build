"""
🎯 Workshop 5: The Complete Transformer
========================================

🛸 THE BRAIN ANALOGY:
Our alien has learned all the components separately:
    - 📝 Tokenization: How to read symbols (Workshop 1)
    - 🗺️ Embeddings: Where words live in meaning-space (Workshop 2)
    - 📚 Vector DB: How to store and retrieve knowledge (Workshop 3)
    - 👀 Attention: How to focus on what matters (Workshop 4)

Now it's time to BUILD THE COMPLETE BRAIN! 🧠

Think of a transformer like a factory assembly line:
    1. Raw materials come in (tokens)
    2. They get enriched with position info (positional encoding)
    3. Workers discuss and share context (attention)
    4. Each item gets individually processed (feed-forward)
    5. Quality checks at every step (layer norm)
    6. The final product emerges (next token prediction)

THIS IS EXACTLY WHAT A TRANSFORMER DOES.
Stack these layers, and you get GPT, Claude, LLaMA!

Usage:
    python transformer.py
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# PART 1: BUILDING BLOCKS (Components we've seen before)
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization - stabilizes training by normalizing each position.
    
    🛸 Analogy: Like a "quality calibrator" that ensures each worker's 
    output is on the same scale, preventing any one voice from dominating.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation - smooth version of ReLU used in transformers.
    
    🛸 Analogy: A "smart gate" that lets positive signals through 
    but softly dampens negative ones (unlike ReLU's hard cutoff).
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# =============================================================================
# PART 2: POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding:
    """
    Adds position information to embeddings using sinusoidal patterns.
    
    🛸 Analogy: Transformers process all words at once (unlike reading 
    left-to-right). So we need to STAMP each word with its position number,
    like seat numbers at a concert!
    
    We use sine/cosine waves at different frequencies - this creates a 
    unique "fingerprint" for each position that the model can learn from.
    """
    
    def __init__(self, embed_dim: int, max_seq_len: int = 512):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.encoding = self._create_encoding()
    
    def _create_encoding(self) -> np.ndarray:
        """Create sinusoidal position encodings."""
        position = np.arange(self.max_seq_len)[:, np.newaxis]
        dim_indices = np.arange(self.embed_dim)[np.newaxis, :]
        
        # Wavelengths range from 2π to 10000·2π
        angles = position / np.power(10000, (2 * (dim_indices // 2)) / self.embed_dim)
        
        # Apply sin to even indices, cos to odd
        encoding = np.zeros((self.max_seq_len, self.embed_dim))
        encoding[:, 0::2] = np.sin(angles[:, 0::2])
        encoding[:, 1::2] = np.cos(angles[:, 1::2])
        
        return encoding.astype(np.float32)
    
    def __call__(self, seq_len: int) -> np.ndarray:
        """Get positional encoding for a sequence of given length."""
        return self.encoding[:seq_len]


# =============================================================================
# PART 3: MULTI-HEAD ATTENTION (From Workshop 4)
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Self-Attention with optional causal masking.
    
    🛸 Analogy: Instead of one spotlight, we have MULTIPLE spotlights
    (heads), each looking for different patterns:
        - Head 1: Tracks grammar (subject-verb)
        - Head 2: Tracks pronouns (it → the dog)
        - Head 3: Tracks nearby words
        - Head 4: Tracks semantic similarity
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Initialize projections (normally learned, here random for demo)
        np.random.seed(42)
        scale = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
    
    def __call__(self, x: np.ndarray, causal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multi-head attention.
        
        Args:
            x: Input of shape (seq_len, embed_dim)
            causal: If True, apply causal mask (for generation)
        
        Returns:
            output: Attended output (seq_len, embed_dim)
            weights: Attention weights (num_heads, seq_len, seq_len)
        """
        seq_len = x.shape[0]
        
        # Project to Q, K, V
        Q = x @ self.W_q  # (seq_len, embed_dim)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape for multi-head: (seq_len, num_heads, head_dim)
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim)
        K = K.reshape(seq_len, self.num_heads, self.head_dim)
        V = V.reshape(seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 0, 2)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)
        
        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / scale  # (num_heads, seq_len, seq_len)
        
        # Apply causal mask if needed
        if causal:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
            scores = scores + mask
        
        # Softmax
        weights = softmax(scores, axis=-1)
        
        # Apply attention to values
        attended = np.matmul(weights, V)  # (num_heads, seq_len, head_dim)
        
        # Reshape back: (seq_len, embed_dim)
        attended = attended.transpose(1, 0, 2).reshape(seq_len, self.embed_dim)
        
        # Final projection
        output = attended @ self.W_o
        
        return output, weights


# =============================================================================
# PART 4: FEED-FORWARD NETWORK
# =============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    🛸 Analogy: After the attention "discussion", each word goes through 
    its own "thinking booth" where it processes what it learned independently.
    
    The expansion to 4x dimensions and back acts like a "memory scratch pad"
    where complex transformations can happen.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim * 4
        
        # Initialize weights
        np.random.seed(43)
        scale = np.sqrt(2.0 / embed_dim)
        self.W1 = np.random.randn(embed_dim, self.hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(self.hidden_dim, embed_dim).astype(np.float32) * scale
        self.b2 = np.zeros(embed_dim, dtype=np.float32)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply feed-forward transformation."""
        # Up-project, activate, down-project
        hidden = gelu(x @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        return output


# =============================================================================
# PART 5: TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock:
    """
    A single transformer layer: Attention + Feed-Forward with residuals.
    
    🛸 Analogy: One "floor" of our alien's brain-factory:
        1. Words discuss with each other (attention)
        2. Each word processes individually (feed-forward)
        3. Original information is preserved (residual connections)
        4. Everything stays calibrated (layer norm)
    
    Architecture:
        x → LayerNorm → Attention → + → LayerNorm → FFN → + → output
            └──────────────────────┘   └────────────────┘
                   residual                 residual
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim)
        self.embed_dim = embed_dim
    
    def __call__(self, x: np.ndarray, causal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply one transformer block.
        
        Returns:
            output: Transformed embeddings
            attn_weights: Attention weights for visualization
        """
        # Pre-norm attention with residual
        normed = layer_norm(x)
        attended, weights = self.attention(normed, causal=causal)
        x = x + attended  # Residual connection
        
        # Pre-norm feed-forward with residual
        normed = layer_norm(x)
        ff_out = self.ffn(normed)
        x = x + ff_out  # Residual connection
        
        return x, weights


# =============================================================================
# PART 6: COMPLETE TRANSFORMER (Mini-GPT!)
# =============================================================================

class MiniTransformer:
    """
    A minimal decoder-only transformer (GPT-style).
    
    🛸 THE COMPLETE ALIEN BRAIN:
    
    Input: "The cat sat on the"
                ↓
    [Token Embeddings] + [Position Encodings]
                ↓
    [Transformer Block 1] - Attention + FFN
                ↓
    [Transformer Block 2] - Attention + FFN
                ↓
           ... (N layers)
                ↓
    [Layer Norm] → [Output Projection]
                ↓
    Output: Probability of each word being next
            → "mat" (highest probability!)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 128
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embedding matrix
        np.random.seed(44)
        self.token_embedding = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Transformer blocks
        self.blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        
        # Output projection (tied with input embeddings for efficiency)
        self.output_proj = self.token_embedding.T  # (embed_dim, vocab_size)
    
    def forward(self, token_ids: List[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the transformer.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            logits: Unnormalized scores for next token (seq_len, vocab_size)
            all_weights: Attention weights from each layer
        """
        seq_len = len(token_ids)
        
        # 1. Token embeddings
        x = self.token_embedding[token_ids]  # (seq_len, embed_dim)
        
        # 2. Add positional encoding
        x = x + self.pos_encoding(seq_len)
        
        # 3. Pass through transformer blocks
        all_weights = []
        for block in self.blocks:
            x, weights = block(x, causal=True)
            all_weights.append(weights)
        
        # 4. Final layer norm
        x = layer_norm(x)
        
        # 5. Project to vocabulary
        logits = x @ self.output_proj  # (seq_len, vocab_size)
        
        return logits, all_weights
    
    def get_next_token_probs(self, token_ids: List[int], temperature: float = 1.0) -> np.ndarray:
        """Get probability distribution for the next token."""
        logits, _ = self.forward(token_ids)
        
        # Get logits for last position
        last_logits = logits[-1] / temperature
        
        # Apply softmax
        probs = softmax(last_logits)
        return probs
    
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 10
    ) -> List[int]:
        """
        Generate new tokens autoregressively.
        
        🛸 This is how ChatGPT generates text:
            1. Take the prompt
            2. Predict next token
            3. Add it to the sequence
            4. Repeat!
        """
        generated = list(prompt_ids)
        
        for _ in range(max_new_tokens):
            # Get probabilities for next token
            probs = self.get_next_token_probs(generated, temperature)
            
            # Top-k sampling: only consider top k tokens
            top_k_indices = np.argsort(probs)[-top_k:]
            top_k_probs = probs[top_k_indices]
            top_k_probs = top_k_probs / top_k_probs.sum()  # Renormalize
            
            # Sample from top-k
            next_token = np.random.choice(top_k_indices, p=top_k_probs)
            generated.append(int(next_token))
            
            # Stop at max length
            if len(generated) >= self.max_seq_len:
                break
        
        return generated


# =============================================================================
# PART 7: SIMPLE TOKENIZER FOR DEMOS
# =============================================================================

class SimpleVocab:
    """A simple word-level vocabulary for demonstration."""
    
    def __init__(self):
        # Common words for our demos
        self.words = [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>",
            "the", "a", "an", "is", "are", "was", "were",
            "cat", "dog", "bird", "fish", "mouse",
            "sat", "ran", "jumped", "flew", "swam",
            "on", "in", "under", "over", "behind",
            "mat", "box", "tree", "house", "table",
            "big", "small", "quick", "lazy", "happy",
            "and", "but", "or", "because", "so",
            "it", "he", "she", "they", "we",
            "chased", "ate", "saw", "caught", "found"
        ]
        self.word_to_id = {w: i for i, w in enumerate(self.words)}
        self.id_to_word = {i: w for i, w in enumerate(self.words)}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        words = text.lower().split()
        return [self.word_to_id.get(w, 1) for w in words]  # 1 = <UNK>
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return " ".join(self.id_to_word.get(i, "<UNK>") for i in ids)
    
    @property
    def vocab_size(self) -> int:
        return len(self.words)


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def demo_transformer():
    """Demonstrate the transformer components."""
    
    print("=" * 70)
    print("🧠 TRANSFORMER DEMO - Mini-GPT in Action!")
    print("=" * 70)
    
    # Create vocabulary and model
    vocab = SimpleVocab()
    model = MiniTransformer(
        vocab_size=vocab.vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2
    )
    
    print(f"\n📊 Model Configuration:")
    print(f"   Vocabulary size: {vocab.vocab_size}")
    print(f"   Embedding dim: {model.embed_dim}")
    print(f"   Attention heads: {model.num_heads}")
    print(f"   Transformer layers: {model.num_layers}")
    
    # Demo 1: Forward pass
    print("\n" + "-" * 70)
    print("📝 DEMO 1: Forward Pass")
    print("-" * 70)
    
    prompt = "the cat sat on the"
    token_ids = vocab.encode(prompt)
    
    print(f"   Input: '{prompt}'")
    print(f"   Token IDs: {token_ids}")
    
    logits, all_weights = model.forward(token_ids)
    print(f"   Output shape: {logits.shape} (seq_len × vocab_size)")
    
    # Get top predictions for next token
    probs = softmax(logits[-1])
    top_5_indices = np.argsort(probs)[-5:][::-1]
    
    print(f"\n   Top 5 next-token predictions:")
    for idx in top_5_indices:
        word = vocab.id_to_word.get(idx, "<UNK>")
        print(f"      '{word}': {probs[idx]:.3f}")
    
    # Demo 2: Text generation
    print("\n" + "-" * 70)
    print("📝 DEMO 2: Text Generation")
    print("-" * 70)
    
    prompts = [
        "the cat",
        "a big dog",
        "the bird flew"
    ]
    
    for prompt in prompts:
        token_ids = vocab.encode(prompt)
        generated_ids = model.generate(
            token_ids,
            max_new_tokens=5,
            temperature=0.8,
            top_k=5
        )
        generated_text = vocab.decode(generated_ids)
        print(f"   Prompt: '{prompt}'")
        print(f"   Generated: '{generated_text}'")
        print()
    
    # Demo 3: Temperature effects
    print("-" * 70)
    print("📝 DEMO 3: Temperature Effects")
    print("-" * 70)
    
    prompt = "the dog"
    token_ids = vocab.encode(prompt)
    
    print(f"   Prompt: '{prompt}'")
    print()
    
    for temp in [0.5, 1.0, 2.0]:
        generated_ids = model.generate(
            token_ids,
            max_new_tokens=5,
            temperature=temp,
            top_k=10
        )
        generated_text = vocab.decode(generated_ids)
        print(f"   Temperature {temp}: '{generated_text}'")
    
    print("\n" + "=" * 70)
    print("✅ Transformer demo complete!")
    print("=" * 70)


def demo_components():
    """Demo individual components."""
    
    print("\n" + "=" * 70)
    print("🔧 COMPONENT DEMOS")
    print("=" * 70)
    
    # Positional encoding
    print("\n📍 Positional Encoding:")
    pos_enc = PositionalEncoding(embed_dim=8, max_seq_len=10)
    enc = pos_enc(5)
    print(f"   Shape: {enc.shape}")
    print(f"   Position 0: {enc[0, :4]}...")
    print(f"   Position 4: {enc[4, :4]}...")
    
    # Multi-head attention
    print("\n👀 Multi-Head Attention:")
    attn = MultiHeadAttention(embed_dim=64, num_heads=4)
    x = np.random.randn(5, 64).astype(np.float32)
    out, weights = attn(x, causal=True)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Weights: {weights.shape} (num_heads × seq × seq)")
    
    # Feed-forward
    print("\n🔄 Feed-Forward Network:")
    ffn = FeedForward(embed_dim=64)
    out = ffn(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   Hidden dim: {ffn.hidden_dim} (4× expansion)")
    
    # Transformer block
    print("\n🧱 Transformer Block:")
    block = TransformerBlock(embed_dim=64, num_heads=4)
    out, weights = block(x, causal=True)
    print(f"   Input: {x.shape}")
    print(f"   Output: {out.shape}")
    print(f"   (Attention + FFN + Residuals + LayerNorm)")


if __name__ == "__main__":
    demo_components()
    print()
    demo_transformer()
