"""
🛸 Shape Data: Zara's Shape World
==================================

Shared data definitions for the "Zara Crash-Lands in Shape World" keynote.
Contains sample shape scenes, named patterns, color palettes, and helpers.

Usage:
    from shape_data import SAMPLE_SCENES, SHAPE_PATTERNS, LAYER_COLORS
"""

import numpy as np
from typing import Dict, List, Any


# =========================================================================
# SHAPE TYPES — Zara's vocabulary of things she sees
# =========================================================================

SHAPE_TYPES = [
    "circle", "rectangle", "triangle", "line", "polygon",
    "diamond", "star", "arc", "ellipse", "cross"
]

SHAPE_TYPE_TO_ID = {s: i for i, s in enumerate(SHAPE_TYPES)}
ID_TO_SHAPE_TYPE = {i: s for s, i in SHAPE_TYPE_TO_ID.items()}

# =========================================================================
# COLORS — What Zara sees when she opens her eyes
# =========================================================================

LAYER_COLORS = {
    "red": "#ef4444",
    "blue": "#3b82f6",
    "green": "#22c55e",
    "purple": "#a855f7",
    "orange": "#f97316",
    "yellow": "#eab308",
    "pink": "#ec4899",
    "teal": "#14b8a6",
    "slate": "#64748b",
    "white": "#f8fafc",
}

# =========================================================================
# SHAPE PARTS — Sub-shape primitives (like BPE tokens for vision)
# =========================================================================

SHAPE_PARTS = [
    "straight_edge", "curve", "sharp_corner", "right_angle",
    "gentle_corner", "fill_solid", "fill_empty", "intersection",
    "parallel_pair", "concentric"
]

PART_TO_ID = {p: i for i, p in enumerate(SHAPE_PARTS)}

# What parts make up each shape type
SHAPE_DECOMPOSITION = {
    "circle": ["curve", "curve", "curve", "curve", "fill_solid"],
    "rectangle": ["straight_edge", "right_angle", "straight_edge", "right_angle",
                   "straight_edge", "right_angle", "straight_edge", "right_angle", "fill_solid"],
    "triangle": ["straight_edge", "sharp_corner", "straight_edge", "sharp_corner",
                  "straight_edge", "sharp_corner", "fill_solid"],
    "line": ["straight_edge"],
    "diamond": ["straight_edge", "sharp_corner", "straight_edge", "sharp_corner",
                "straight_edge", "sharp_corner", "straight_edge", "sharp_corner", "fill_solid"],
    "star": ["sharp_corner", "straight_edge"] * 5 + ["fill_solid"],
    "arc": ["curve", "fill_empty"],
    "ellipse": ["curve", "curve", "gentle_corner", "gentle_corner", "fill_solid"],
    "cross": ["straight_edge", "right_angle"] * 4 + ["intersection", "fill_solid"],
    "polygon": ["straight_edge", "gentle_corner"] * 3 + ["fill_solid"],
}


# =========================================================================
# SAMPLE SCENES — What Zara sees when she opens her eyes
# =========================================================================

def _make_shape(shape_type: str, x: float, y: float, w: float, h: float,
                color: str, label: str = "") -> Dict[str, Any]:
    """Create a shape dictionary."""
    return {
        "type": shape_type,
        "x": x, "y": y, "w": w, "h": h,
        "color": color,
        "label": label or shape_type,
    }


SAMPLE_SCENES = {
    "house": {
        "name": "House",
        "shapes": [
            _make_shape("rectangle", 2, 3, 6, 5, "blue", "wall"),
            _make_shape("triangle", 1, 3, 8, 3, "red", "roof"),
            _make_shape("rectangle", 4, 5, 2, 3, "orange", "door"),
            _make_shape("rectangle", 2.5, 4, 1.5, 1.5, "yellow", "window_L"),
            _make_shape("rectangle", 6, 4, 1.5, 1.5, "yellow", "window_R"),
        ]
    },
    "tree": {
        "name": "Tree",
        "shapes": [
            _make_shape("rectangle", 4, 5, 2, 4, "orange", "trunk"),
            _make_shape("circle", 2.5, 2, 5, 5, "green", "canopy_bottom"),
            _make_shape("circle", 3, 0.5, 4, 4, "green", "canopy_top"),
        ]
    },
    "face": {
        "name": "Face",
        "shapes": [
            _make_shape("circle", 1, 1, 8, 8, "yellow", "head"),
            _make_shape("circle", 2.5, 3, 1.5, 1.5, "white", "eye_L"),
            _make_shape("circle", 6, 3, 1.5, 1.5, "white", "eye_R"),
            _make_shape("arc", 3, 6, 4, 2, "red", "mouth"),
        ]
    },
    "car": {
        "name": "Car",
        "shapes": [
            _make_shape("rectangle", 1, 3, 8, 3, "blue", "body"),
            _make_shape("rectangle", 2.5, 1, 5, 2.5, "blue", "cabin"),
            _make_shape("circle", 2.5, 6, 1.5, 1.5, "slate", "wheel_L"),
            _make_shape("circle", 7, 6, 1.5, 1.5, "slate", "wheel_R"),
            _make_shape("rectangle", 3, 1.5, 2, 1.5, "teal", "window_front"),
            _make_shape("rectangle", 5.5, 1.5, 2, 1.5, "teal", "window_rear"),
        ]
    },
    "rocket": {
        "name": "Rocket",
        "shapes": [
            _make_shape("rectangle", 3.5, 2, 3, 6, "slate", "fuselage"),
            _make_shape("triangle", 3.5, 0, 3, 2, "red", "nose_cone"),
            _make_shape("triangle", 2, 6, 2, 3, "orange", "fin_L"),
            _make_shape("triangle", 6, 6, 2, 3, "orange", "fin_R"),
            _make_shape("circle", 4.5, 3.5, 1, 1, "blue", "porthole"),
        ]
    },
    "snowman": {
        "name": "Snowman",
        "shapes": [
            _make_shape("circle", 2.5, 5, 5, 5, "white", "base"),
            _make_shape("circle", 3, 2.5, 4, 4, "white", "body"),
            _make_shape("circle", 3.5, 0.5, 3, 3, "white", "head"),
            _make_shape("triangle", 5, 1.5, 1.5, 1, "orange", "nose"),
        ]
    },
    "butterfly": {
        "name": "Butterfly",
        "shapes": [
            _make_shape("ellipse", 4.5, 3, 1, 4, "slate", "body"),
            _make_shape("ellipse", 1, 1, 4, 3.5, "purple", "wing_UL"),
            _make_shape("ellipse", 5, 1, 4, 3.5, "purple", "wing_UR"),
            _make_shape("ellipse", 1.5, 4, 3.5, 3, "pink", "wing_LL"),
            _make_shape("ellipse", 5.5, 4, 3.5, 3, "pink", "wing_LR"),
        ]
    },
    "robot": {
        "name": "Robot",
        "shapes": [
            _make_shape("rectangle", 3, 0, 4, 3, "slate", "head"),
            _make_shape("rectangle", 2, 3, 6, 5, "blue", "torso"),
            _make_shape("rectangle", 0.5, 3.5, 1.5, 4, "teal", "arm_L"),
            _make_shape("rectangle", 8, 3.5, 1.5, 4, "teal", "arm_R"),
            _make_shape("rectangle", 3, 8, 1.5, 3, "slate", "leg_L"),
            _make_shape("rectangle", 5.5, 8, 1.5, 3, "slate", "leg_R"),
            _make_shape("circle", 3.8, 0.8, 1, 1, "red", "eye_L"),
            _make_shape("circle", 5.2, 0.8, 1, 1, "red", "eye_R"),
        ]
    },
    "boat": {
        "name": "Boat",
        "shapes": [
            _make_shape("polygon", 1, 5, 8, 3, "blue", "hull"),
            _make_shape("rectangle", 4.5, 1, 1, 4, "orange", "mast"),
            _make_shape("triangle", 5.5, 1, 3, 4, "white", "sail"),
        ]
    },
    "crown": {
        "name": "Crown",
        "shapes": [
            _make_shape("rectangle", 2, 4, 6, 3, "yellow", "band"),
            _make_shape("triangle", 2, 2, 2, 2, "yellow", "point_L"),
            _make_shape("triangle", 4, 1, 2, 3, "yellow", "point_C"),
            _make_shape("triangle", 6, 2, 2, 2, "yellow", "point_R"),
            _make_shape("circle", 3, 4.5, 0.8, 0.8, "red", "jewel_L"),
            _make_shape("circle", 5, 4.5, 0.8, 0.8, "blue", "jewel_C"),
            _make_shape("circle", 7, 4.5, 0.8, 0.8, "green", "jewel_R"),
        ]
    },
    "flower": {
        "name": "Flower",
        "shapes": [
            _make_shape("rectangle", 4.5, 5, 1, 4, "green", "stem"),
            _make_shape("circle", 3.5, 3, 3, 3, "yellow", "center"),
            _make_shape("ellipse", 2, 1, 2, 2, "pink", "petal_TL"),
            _make_shape("ellipse", 6, 1, 2, 2, "pink", "petal_TR"),
            _make_shape("ellipse", 1, 3.5, 2, 2, "pink", "petal_L"),
            _make_shape("ellipse", 7, 3.5, 2, 2, "pink", "petal_R"),
            _make_shape("ellipse", 2, 5.5, 2, 2, "pink", "petal_BL"),
            _make_shape("ellipse", 6, 5.5, 2, 2, "pink", "petal_BR"),
        ]
    },
    "fish": {
        "name": "Fish",
        "shapes": [
            _make_shape("ellipse", 2, 2.5, 6, 4, "blue", "body"),
            _make_shape("triangle", 7.5, 2, 2.5, 5, "teal", "tail"),
            _make_shape("circle", 3, 3.5, 1, 1, "white", "eye"),
        ]
    },
    # --- Symmetric patterns (for attention demo) ---
    "symmetric_triangles": {
        "name": "Symmetric Triangles",
        "shapes": [
            _make_shape("triangle", 1, 2, 3, 4, "red", "tri_L"),
            _make_shape("triangle", 6, 2, 3, 4, "red", "tri_R"),
            _make_shape("line", 4.5, 0, 1, 9, "slate", "axis"),
        ]
    },
    "nested_circles": {
        "name": "Nested Circles",
        "shapes": [
            _make_shape("circle", 1, 1, 8, 8, "blue", "outer"),
            _make_shape("circle", 2.5, 2.5, 5, 5, "green", "middle"),
            _make_shape("circle", 4, 4, 2, 2, "red", "inner"),
        ]
    },
    "grid_pattern": {
        "name": "Grid Pattern",
        "shapes": [
            _make_shape("rectangle", 0.5, 0.5, 2, 2, "red", "cell_00"),
            _make_shape("rectangle", 3.5, 0.5, 2, 2, "blue", "cell_10"),
            _make_shape("rectangle", 6.5, 0.5, 2, 2, "red", "cell_20"),
            _make_shape("rectangle", 0.5, 3.5, 2, 2, "blue", "cell_01"),
            _make_shape("rectangle", 3.5, 3.5, 2, 2, "red", "cell_11"),
            _make_shape("rectangle", 6.5, 3.5, 2, 2, "blue", "cell_21"),
            _make_shape("rectangle", 0.5, 6.5, 2, 2, "red", "cell_02"),
            _make_shape("rectangle", 3.5, 6.5, 2, 2, "blue", "cell_12"),
            _make_shape("rectangle", 6.5, 6.5, 2, 2, "red", "cell_22"),
        ]
    },
    # --- Layout-like patterns (for the reveal) ---
    "diff_pair_hint": {
        "name": "Matched Pair",
        "shapes": [
            _make_shape("rectangle", 1, 2, 3, 4, "green", "device_L"),
            _make_shape("rectangle", 6, 2, 3, 4, "green", "device_R"),
            _make_shape("line", 1, 1, 8, 0.5, "blue", "wire_top"),
            _make_shape("line", 1, 7, 8, 0.5, "blue", "wire_bottom"),
            _make_shape("rectangle", 4, 3, 2, 2, "red", "shared_node"),
        ]
    },
    "layout_hint": {
        "name": "Layout Preview",
        "shapes": [
            _make_shape("rectangle", 0.5, 0.5, 3, 2, "green", "block_A"),
            _make_shape("rectangle", 4, 0.5, 3, 2, "green", "block_B"),
            _make_shape("rectangle", 7.5, 0.5, 2, 2, "purple", "block_C"),
            _make_shape("line", 2, 2.5, 0.5, 2, "blue", "via_1"),
            _make_shape("line", 5.5, 2.5, 0.5, 2, "blue", "via_2"),
            _make_shape("rectangle", 0.5, 4.5, 9, 0.5, "blue", "metal_bus"),
            _make_shape("rectangle", 1, 5.5, 3, 2, "teal", "block_D"),
            _make_shape("rectangle", 5, 5.5, 4, 2, "orange", "block_E"),
            _make_shape("rectangle", 0.5, 8, 9, 0.5, "purple", "metal_bus_2"),
        ]
    },
}


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def scene_to_feature_vector(scene: Dict[str, Any], dims: int = 16) -> np.ndarray:
    """
    Convert a shape scene into a fixed-length feature vector for embeddings/search.

    Encodes: shape type counts, average position, color diversity, shape count.
    """
    shapes = scene["shapes"]
    vec = np.zeros(dims)

    # Feature 0-9: shape type counts (normalized)
    for s in shapes:
        sid = SHAPE_TYPE_TO_ID.get(s["type"], 0)
        if sid < 10:
            vec[sid] += 1
    total = max(len(shapes), 1)
    vec[:10] /= total

    # Feature 10-11: average center position (normalized to 0-1 assuming 10x10 canvas)
    if shapes:
        vec[10] = np.mean([(s["x"] + s["w"] / 2) / 10 for s in shapes])
        vec[11] = np.mean([(s["y"] + s["h"] / 2) / 10 for s in shapes])

    # Feature 12: number of shapes (normalized)
    vec[12] = len(shapes) / 10.0

    # Feature 13: number of unique colors (normalized)
    vec[13] = len(set(s["color"] for s in shapes)) / len(LAYER_COLORS)

    # Feature 14: spatial spread (std of positions)
    if len(shapes) > 1:
        xs = [(s["x"] + s["w"] / 2) for s in shapes]
        ys = [(s["y"] + s["h"] / 2) for s in shapes]
        vec[14] = (np.std(xs) + np.std(ys)) / 10.0

    # Feature 15: symmetry score (rough — compare left vs right half)
    if shapes:
        left = sum(1 for s in shapes if s["x"] + s["w"] / 2 < 5)
        right = sum(1 for s in shapes if s["x"] + s["w"] / 2 >= 5)
        vec[15] = 1.0 - abs(left - right) / total

    return vec


def get_all_pattern_names() -> List[str]:
    """Return sorted list of all available pattern names."""
    return sorted(SAMPLE_SCENES.keys())


def get_scene(name: str) -> Dict[str, Any]:
    """Get a scene by name."""
    return SAMPLE_SCENES[name]
