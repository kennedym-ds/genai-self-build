"""
🛸 Shapes Core: Zara's Tools for Understanding Shape World
============================================================

All four demo engines in one file:
  1. ShapeTokenizer  — decompose scenes into parts (pixel / shape / part-level)
  2. ShapeEmbedding  — map shapes into a meaning space where similar = nearby
  3. ShapeVectorDB   — searchable library of shape patterns
  4. ShapeAttention   — multi-head attention showing which shapes relate

These power the live demo moments in the keynote talk.

Usage:
    from shapes_core import ShapeTokenizer, ShapeEmbedding, ShapeVectorDB, ShapeAttention
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

from shape_data import (
    SHAPE_TYPES, SHAPE_TYPE_TO_ID, ID_TO_SHAPE_TYPE,
    SHAPE_PARTS, PART_TO_ID, SHAPE_DECOMPOSITION,
    SAMPLE_SCENES, LAYER_COLORS, scene_to_feature_vector,
)


# =========================================================================
# PART 1: SHAPE TOKENIZER — "What Am I Looking At?"
# =========================================================================

class ShapeTokenizer:
    """
    Zara's first tool: breaking a visual scene into manageable pieces.

    Three strategies mirror text tokenization:
      - 'pixel':   rasterize to a grid (like char-level — huge, complete)
      - 'shape':   catalog whole shapes (like word-level — compact, brittle)
      - 'part':    decompose into sub-shape parts (like BPE — smart, flexible)

    Example:
        tok = ShapeTokenizer(strategy='part')
        tok.train(list(SAMPLE_SCENES.values()))
        tokens = tok.encode(SAMPLE_SCENES['house'])
        # [0, 3, 0, 3, 0, 3, 0, 3, 5, ...]  (part IDs)
    """

    def __init__(self, strategy: str = 'part'):
        if strategy not in ('pixel', 'shape', 'part'):
            raise ValueError(f"Strategy must be 'pixel', 'shape', or 'part', got '{strategy}'")
        self.strategy = strategy
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.is_trained = False

    def train(self, scenes: List[Dict[str, Any]]) -> None:
        """Build vocabulary from a list of scenes."""
        if self.strategy == 'pixel':
            self._train_pixel()
        elif self.strategy == 'shape':
            self._train_shape(scenes)
        else:
            self._train_part()
        self.is_trained = True

    def _train_pixel(self) -> None:
        """Pixel strategy: vocabulary is a finite set of color values."""
        colors = ["empty"] + list(LAYER_COLORS.keys())
        self.vocab = {c: i for i, c in enumerate(colors)}
        self.inverse_vocab = {i: c for c, i in self.vocab.items()}

    def _train_shape(self, scenes: List[Dict[str, Any]]) -> None:
        """Shape strategy: every unique shape-label becomes a token."""
        self.vocab = {"<UNK>": 0}
        idx = 1
        for scene in scenes:
            for s in scene.get("shapes", []):
                key = f'{s["type"]}_{s["color"]}'
                if key not in self.vocab:
                    self.vocab[key] = idx
                    idx += 1
        self.inverse_vocab = {i: t for t, i in self.vocab.items()}

    def _train_part(self) -> None:
        """Part strategy: vocabulary is the set of sub-shape primitives."""
        self.vocab = {p: i for i, p in enumerate(SHAPE_PARTS)}
        self.inverse_vocab = {i: p for p, i in self.vocab.items()}

    def encode(self, scene: Dict[str, Any]) -> List[int]:
        """Encode a scene into a list of token IDs."""
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")

        if self.strategy == 'pixel':
            return self._encode_pixel(scene)
        elif self.strategy == 'shape':
            return self._encode_shape(scene)
        else:
            return self._encode_part(scene)

    def _encode_pixel(self, scene: Dict[str, Any], grid_size: int = 10) -> List[int]:
        """Rasterize scene to a grid. Each cell gets the color of whatever shape covers it."""
        grid = [[0] * grid_size for _ in range(grid_size)]
        empty_id = self.vocab["empty"]
        for s in scene.get("shapes", []):
            color_id = self.vocab.get(s["color"], empty_id)
            x0, y0 = int(s["x"]), int(s["y"])
            x1 = min(x0 + int(np.ceil(s["w"])), grid_size)
            y1 = min(y0 + int(np.ceil(s["h"])), grid_size)
            for r in range(max(y0, 0), y1):
                for c in range(max(x0, 0), x1):
                    grid[r][c] = color_id
        return [cell for row in grid for cell in row]

    def _encode_shape(self, scene: Dict[str, Any]) -> List[int]:
        """One token per shape."""
        unk = self.vocab.get("<UNK>", 0)
        tokens = []
        for s in scene.get("shapes", []):
            key = f'{s["type"]}_{s["color"]}'
            tokens.append(self.vocab.get(key, unk))
        return tokens

    def _encode_part(self, scene: Dict[str, Any]) -> List[int]:
        """Decompose each shape into sub-parts."""
        tokens = []
        for s in scene.get("shapes", []):
            parts = SHAPE_DECOMPOSITION.get(s["type"], ["straight_edge"])
            for p in parts:
                tokens.append(self.vocab.get(p, 0))
        return tokens

    def decode(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs back to human-readable names."""
        return [self.inverse_vocab.get(tid, "?") for tid in token_ids]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# =========================================================================
# PART 2: SHAPE EMBEDDING — "The Meaning Map"
# =========================================================================

class ShapeEmbedding:
    """
    Zara's meaning map: shapes that appear together get similar vectors.

    Three strategies:
      - 'random':         random vectors (baseline — no real meaning)
      - 'spatial':        co-occurrence based (shapes near each other get similar vectors)
      - 'feature':        feature-engineered vectors from scene properties

    Example:
        emb = ShapeEmbedding(strategy='spatial', dimensions=16)
        emb.train(list(SAMPLE_SCENES.values()))
        emb.most_similar('house')  # [('car', 0.82), ('boat', 0.71), ...]
    """

    def __init__(self, strategy: str = 'spatial', dimensions: int = 16):
        if strategy not in ('random', 'spatial', 'feature'):
            raise ValueError(f"Strategy must be 'random', 'spatial', or 'feature', got '{strategy}'")
        self.strategy = strategy
        self.dimensions = dimensions
        self.scene_names: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_trained = False

    def train(self, scenes: Dict[str, Dict[str, Any]]) -> None:
        """
        Train embeddings from a dict of named scenes.

        Args:
            scenes: dict mapping scene name -> scene dict (like SAMPLE_SCENES)
        """
        self.scene_names = list(scenes.keys())
        n = len(self.scene_names)

        if self.strategy == 'random':
            np.random.seed(42)
            self.embeddings = np.random.randn(n, self.dimensions).astype(np.float32)
        elif self.strategy == 'feature':
            self.embeddings = np.array([
                scene_to_feature_vector(scenes[name], self.dimensions)
                for name in self.scene_names
            ], dtype=np.float32)
        else:  # spatial
            self._train_spatial(scenes)

        # Normalize rows
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms
        self.is_trained = True

    def _train_spatial(self, scenes: Dict[str, Dict[str, Any]]) -> None:
        """
        Build co-occurrence-style embeddings: scenes that share shape types
        and color patterns get similar vectors.
        """
        n = len(self.scene_names)
        # Build a feature matrix from shape-type and color distributions
        raw = np.zeros((n, len(SHAPE_TYPES) + len(LAYER_COLORS) + 4), dtype=np.float32)
        for i, name in enumerate(self.scene_names):
            shapes = scenes[name]["shapes"]
            for s in shapes:
                sid = SHAPE_TYPE_TO_ID.get(s["type"], 0)
                raw[i, sid] += 1
                cidx = list(LAYER_COLORS.keys()).index(s["color"]) if s["color"] in LAYER_COLORS else 0
                raw[i, len(SHAPE_TYPES) + cidx] += 1
            # Add spatial features
            if shapes:
                xs = [s["x"] + s["w"] / 2 for s in shapes]
                ys = [s["y"] + s["h"] / 2 for s in shapes]
                raw[i, -4] = np.mean(xs) / 10
                raw[i, -3] = np.mean(ys) / 10
                raw[i, -2] = len(shapes) / 10
                raw[i, -1] = len(set(s["type"] for s in shapes)) / len(SHAPE_TYPES)

        # Reduce to target dimensions via random projection (lightweight PCA alternative)
        np.random.seed(42)
        d = raw.shape[1]
        proj = np.random.randn(d, self.dimensions).astype(np.float32) / np.sqrt(d)
        self.embeddings = raw @ proj

    def get_embedding(self, name: str) -> np.ndarray:
        """Get the embedding vector for a named scene."""
        idx = self.scene_names.index(name)
        return self.embeddings[idx]

    def most_similar(self, name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar scenes to a given one."""
        vec = self.get_embedding(name)
        scores = self.embeddings @ vec  # cosine similarity (already normalized)
        ranked = np.argsort(-scores)
        results = []
        for idx in ranked:
            scene_name = self.scene_names[idx]
            if scene_name != name:
                results.append((scene_name, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def get_2d_projection(self) -> np.ndarray:
        """Project embeddings to 2D for visualization (simple PCA)."""
        centered = self.embeddings - self.embeddings.mean(axis=0)
        # SVD-based 2D projection
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ Vt[:2].T


# =========================================================================
# PART 3: SHAPE VECTOR DB — "The Shape Library"
# =========================================================================

class ShapeVectorDB:
    """
    Zara's searchable pattern library.

    Stores scene embeddings and supports similarity search.

    Example:
        db = ShapeVectorDB(dimensions=16)
        db.add('house', house_vector)
        results = db.search(query_vector, top_k=3)
        # [('house', 0.91), ('car', 0.78), ...]
    """

    def __init__(self, dimensions: int = 16):
        self.dimensions = dimensions
        self.vectors: Dict[str, np.ndarray] = {}
        self.id_list: List[str] = []

    def add(self, name: str, vector: np.ndarray) -> None:
        """Add a pattern to the library."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self.vectors[name] = vector.copy()
        if name not in self.id_list:
            self.id_list.append(name)

    def add_batch(self, names: List[str], vectors: np.ndarray) -> None:
        """Add multiple patterns at once."""
        for name, vec in zip(names, vectors):
            self.add(name, vec)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar patterns to a query vector."""
        if not self.id_list:
            return []
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        mat = np.array([self.vectors[n] for n in self.id_list])
        scores = mat @ query
        ranked = np.argsort(-scores)
        results = []
        for idx in ranked[:top_k]:
            results.append((self.id_list[idx], float(scores[idx])))
        return results

    @property
    def size(self) -> int:
        return len(self.id_list)


# =========================================================================
# PART 4: SHAPE ATTENTION — "The Spotlight"
# =========================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax: turn scores into probabilities."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ShapeAttention:
    """
    Zara's spotlight system: which shapes should pay attention to each other?

    Four attention heads, each looking for a different relationship:
      - Head 0 (Proximity):    shapes that are physically close
      - Head 1 (Color):        shapes that share a color
      - Head 2 (Alignment):    shapes that are horizontally/vertically aligned
      - Head 3 (Containment):  shapes where one is inside another

    Example:
        att = ShapeAttention()
        weights = att.compute_attention(SAMPLE_SCENES['house'])
        # weights shape: (4, num_shapes, num_shapes)
    """

    HEAD_NAMES = ["Proximity", "Color Match", "Alignment", "Containment"]

    def __init__(self):
        pass

    def compute_attention(self, scene: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute multi-head attention weights for all shapes in a scene.

        Returns:
            weights: array of shape (4, n, n) — attention weights per head
            labels:  list of shape labels
        """
        shapes = scene["shapes"]
        n = len(shapes)
        labels = [s["label"] for s in shapes]

        if n == 0:
            return np.zeros((4, 0, 0)), []

        # Compute raw score matrices per head
        proximity = self._proximity_scores(shapes)
        color = self._color_scores(shapes)
        alignment = self._alignment_scores(shapes)
        containment = self._containment_scores(shapes)

        # Stack and apply softmax per row to get attention distributions
        raw = np.stack([proximity, color, alignment, containment])  # (4, n, n)
        weights = np.zeros_like(raw)
        for h in range(4):
            weights[h] = softmax(raw[h], axis=-1)

        return weights, labels

    def _center(self, s: Dict) -> Tuple[float, float]:
        return s["x"] + s["w"] / 2, s["y"] + s["h"] / 2

    def _proximity_scores(self, shapes: List[Dict]) -> np.ndarray:
        """Closer shapes get higher scores."""
        n = len(shapes)
        scores = np.zeros((n, n))
        for i in range(n):
            cx_i, cy_i = self._center(shapes[i])
            for j in range(n):
                if i == j:
                    continue
                cx_j, cy_j = self._center(shapes[j])
                dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
                scores[i, j] = max(0, 10 - dist)  # inverse distance, clamped
        return scores

    def _color_scores(self, shapes: List[Dict]) -> np.ndarray:
        """Same-color shapes get high scores."""
        n = len(shapes)
        scores = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if shapes[i]["color"] == shapes[j]["color"]:
                    scores[i, j] = 5.0
                else:
                    scores[i, j] = 0.5
        return scores

    def _alignment_scores(self, shapes: List[Dict]) -> np.ndarray:
        """Shapes that share x-center or y-center get high scores."""
        n = len(shapes)
        scores = np.zeros((n, n))
        for i in range(n):
            cx_i, cy_i = self._center(shapes[i])
            for j in range(n):
                if i == j:
                    continue
                cx_j, cy_j = self._center(shapes[j])
                x_align = max(0, 2 - abs(cx_i - cx_j))
                y_align = max(0, 2 - abs(cy_i - cy_j))
                scores[i, j] = x_align + y_align
        return scores

    def _containment_scores(self, shapes: List[Dict]) -> np.ndarray:
        """Shape j inside shape i gets high score."""
        n = len(shapes)
        scores = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if center of j is inside bounding box of i
                cx_j, cy_j = self._center(shapes[j])
                if (shapes[i]["x"] <= cx_j <= shapes[i]["x"] + shapes[i]["w"] and
                        shapes[i]["y"] <= cy_j <= shapes[i]["y"] + shapes[i]["h"]):
                    scores[i, j] = 5.0
                else:
                    scores[i, j] = 0.2
        return scores

    def combined_attention(self, scene: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Average attention across all heads."""
        weights, labels = self.compute_attention(scene)
        if weights.size == 0:
            return np.zeros((0, 0)), labels
        combined = weights.mean(axis=0)
        return combined, labels
