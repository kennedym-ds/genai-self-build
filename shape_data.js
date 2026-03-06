// Shape Data — Global constants for the GenAI Self-Build SPA
const SHAPE_TYPES = [
    "circle", "rectangle", "triangle", "line", "polygon",
    "diamond", "star", "arc", "ellipse", "cross"
];

const SHAPE_TYPE_TO_ID = Object.fromEntries(SHAPE_TYPES.map((s, i) => [s, i]));

const LAYER_COLORS = {
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
};

const SHAPE_PARTS = [
    "straight_edge", "curve", "sharp_corner", "right_angle",
    "gentle_corner", "fill_solid", "fill_empty", "intersection",
    "parallel_pair", "concentric"
];

const PART_TO_ID = Object.fromEntries(SHAPE_PARTS.map((p, i) => [p, i]));

const SHAPE_DECOMPOSITION = {
    "circle": ["curve", "curve", "curve", "curve", "fill_solid"],
    "rectangle": ["straight_edge", "right_angle", "straight_edge", "right_angle",
                   "straight_edge", "right_angle", "straight_edge", "right_angle", "fill_solid"],
    "triangle": ["straight_edge", "sharp_corner", "straight_edge", "sharp_corner",
                  "straight_edge", "sharp_corner", "fill_solid"],
    "line": ["straight_edge"],
    "diamond": ["straight_edge", "sharp_corner", "straight_edge", "sharp_corner",
                "straight_edge", "sharp_corner", "straight_edge", "sharp_corner", "fill_solid"],
    "star": Array(5).fill(["sharp_corner", "straight_edge"]).flat().concat(["fill_solid"]),
    "arc": ["curve", "fill_empty"],
    "ellipse": ["curve", "curve", "gentle_corner", "gentle_corner", "fill_solid"],
    "cross": Array(4).fill(["straight_edge", "right_angle"]).flat().concat(["intersection", "fill_solid"]),
    "polygon": Array(3).fill(["straight_edge", "gentle_corner"]).flat().concat(["fill_solid"]),
};

function makeShape(type, x, y, w, h, color, label) {
    return { type, x, y, w, h, color, label: label || type };
}

const SAMPLE_SCENES = {
    "house": {
        name: "House",
        shapes: [
            makeShape("rectangle", 2, 3, 6, 5, "blue", "wall"),
            makeShape("triangle", 1, 3, 8, 3, "red", "roof"),
            makeShape("rectangle", 4, 5, 2, 3, "orange", "door"),
            makeShape("rectangle", 2.5, 4, 1.5, 1.5, "yellow", "window_L"),
            makeShape("rectangle", 6, 4, 1.5, 1.5, "yellow", "window_R"),
        ]
    },
    "tree": {
        name: "Tree",
        shapes: [
            makeShape("rectangle", 4, 5, 2, 4, "orange", "trunk"),
            makeShape("circle", 2.5, 2, 5, 5, "green", "canopy_bottom"),
            makeShape("circle", 3, 0.5, 4, 4, "green", "canopy_top"),
        ]
    },
    "face": {
        name: "Face",
        shapes: [
            makeShape("circle", 1, 1, 8, 8, "yellow", "head"),
            makeShape("circle", 2.5, 3, 1.5, 1.5, "white", "eye_L"),
            makeShape("circle", 6, 3, 1.5, 1.5, "white", "eye_R"),
            makeShape("arc", 3, 6, 4, 2, "red", "mouth"),
        ]
    },
    "car": {
        name: "Car",
        shapes: [
            makeShape("rectangle", 1, 3, 8, 3, "blue", "body"),
            makeShape("rectangle", 2.5, 1, 5, 2.5, "blue", "cabin"),
            makeShape("circle", 2.5, 6, 1.5, 1.5, "slate", "wheel_L"),
            makeShape("circle", 7, 6, 1.5, 1.5, "slate", "wheel_R"),
            makeShape("rectangle", 3, 1.5, 2, 1.5, "teal", "window_front"),
            makeShape("rectangle", 5.5, 1.5, 2, 1.5, "teal", "window_rear"),
        ]
    },
    "rocket": {
        name: "Rocket",
        shapes: [
            makeShape("rectangle", 3.5, 2, 3, 6, "slate", "fuselage"),
            makeShape("triangle", 3.5, 0, 3, 2, "red", "nose_cone"),
            makeShape("triangle", 2, 6, 2, 3, "orange", "fin_L"),
            makeShape("triangle", 6, 6, 2, 3, "orange", "fin_R"),
            makeShape("circle", 4.5, 3.5, 1, 1, "blue", "porthole"),
        ]
    },
    "snowman": {
        name: "Snowman",
        shapes: [
            makeShape("circle", 2.5, 5, 5, 5, "white", "base"),
            makeShape("circle", 3, 2.5, 4, 4, "white", "body"),
            makeShape("circle", 3.5, 0.5, 3, 3, "white", "head"),
            makeShape("triangle", 5, 1.5, 1.5, 1, "orange", "nose"),
        ]
    },
    "butterfly": {
        name: "Butterfly",
        shapes: [
            makeShape("ellipse", 4.5, 3, 1, 4, "slate", "body"),
            makeShape("ellipse", 1, 1, 4, 3.5, "purple", "wing_UL"),
            makeShape("ellipse", 5, 1, 4, 3.5, "purple", "wing_UR"),
            makeShape("ellipse", 1.5, 4, 3.5, 3, "pink", "wing_LL"),
            makeShape("ellipse", 5.5, 4, 3.5, 3, "pink", "wing_LR"),
        ]
    },
    "robot": {
        name: "Robot",
        shapes: [
            makeShape("rectangle", 3, 0, 4, 3, "slate", "head"),
            makeShape("rectangle", 2, 3, 6, 5, "blue", "torso"),
            makeShape("rectangle", 0.5, 3.5, 1.5, 4, "teal", "arm_L"),
            makeShape("rectangle", 8, 3.5, 1.5, 4, "teal", "arm_R"),
            makeShape("rectangle", 3, 8, 1.5, 3, "slate", "leg_L"),
            makeShape("rectangle", 5.5, 8, 1.5, 3, "slate", "leg_R"),
            makeShape("circle", 3.8, 0.8, 1, 1, "red", "eye_L"),
            makeShape("circle", 5.2, 0.8, 1, 1, "red", "eye_R"),
        ]
    },
    "boat": {
        name: "Boat",
        shapes: [
            makeShape("polygon", 1, 5, 8, 3, "blue", "hull"),
            makeShape("rectangle", 4.5, 1, 1, 4, "orange", "mast"),
            makeShape("triangle", 5.5, 1, 3, 4, "white", "sail"),
        ]
    },
    "crown": {
        name: "Crown",
        shapes: [
            makeShape("rectangle", 2, 4, 6, 3, "yellow", "band"),
            makeShape("triangle", 2, 2, 2, 2, "yellow", "point_L"),
            makeShape("triangle", 4, 1, 2, 3, "yellow", "point_C"),
            makeShape("triangle", 6, 2, 2, 2, "yellow", "point_R"),
            makeShape("circle", 3, 4.5, 0.8, 0.8, "red", "jewel_L"),
            makeShape("circle", 5, 4.5, 0.8, 0.8, "blue", "jewel_C"),
            makeShape("circle", 7, 4.5, 0.8, 0.8, "green", "jewel_R"),
        ]
    },
    "flower": {
        name: "Flower",
        shapes: [
            makeShape("rectangle", 4.5, 5, 1, 4, "green", "stem"),
            makeShape("circle", 3.5, 3, 3, 3, "yellow", "center"),
            makeShape("ellipse", 2, 1, 2, 2, "pink", "petal_TL"),
            makeShape("ellipse", 6, 1, 2, 2, "pink", "petal_TR"),
            makeShape("ellipse", 1, 3.5, 2, 2, "pink", "petal_L"),
            makeShape("ellipse", 7, 3.5, 2, 2, "pink", "petal_R"),
            makeShape("ellipse", 2, 5.5, 2, 2, "pink", "petal_BL"),
            makeShape("ellipse", 6, 5.5, 2, 2, "pink", "petal_BR"),
        ]
    },
    "fish": {
        name: "Fish",
        shapes: [
            makeShape("ellipse", 2, 2.5, 6, 4, "blue", "body"),
            makeShape("triangle", 7.5, 2, 2.5, 5, "teal", "tail"),
            makeShape("circle", 3, 3.5, 1, 1, "white", "eye"),
        ]
    },
    "symmetric_triangles": {
        name: "Symmetric Triangles",
        shapes: [
            makeShape("triangle", 1, 2, 3, 4, "red", "tri_L"),
            makeShape("triangle", 6, 2, 3, 4, "red", "tri_R"),
            makeShape("line", 4.5, 0, 1, 9, "slate", "axis"),
        ]
    },
    "nested_circles": {
        name: "Nested Circles",
        shapes: [
            makeShape("circle", 1, 1, 8, 8, "blue", "outer"),
            makeShape("circle", 2.5, 2.5, 5, 5, "green", "middle"),
            makeShape("circle", 4, 4, 2, 2, "red", "inner"),
        ]
    },
    "grid_pattern": {
        name: "Grid Pattern",
        shapes: [
            makeShape("rectangle", 0.5, 0.5, 2, 2, "red", "cell_00"),
            makeShape("rectangle", 3.5, 0.5, 2, 2, "blue", "cell_10"),
            makeShape("rectangle", 6.5, 0.5, 2, 2, "red", "cell_20"),
            makeShape("rectangle", 0.5, 3.5, 2, 2, "blue", "cell_01"),
            makeShape("rectangle", 3.5, 3.5, 2, 2, "red", "cell_11"),
            makeShape("rectangle", 6.5, 3.5, 2, 2, "blue", "cell_21"),
            makeShape("rectangle", 0.5, 6.5, 2, 2, "red", "cell_02"),
            makeShape("rectangle", 3.5, 6.5, 2, 2, "blue", "cell_12"),
            makeShape("rectangle", 6.5, 6.5, 2, 2, "red", "cell_22"),
        ]
    },
    "diff_pair_hint": {
        name: "Matched Pair",
        shapes: [
            makeShape("rectangle", 1, 2, 3, 4, "green", "device_L"),
            makeShape("rectangle", 6, 2, 3, 4, "green", "device_R"),
            makeShape("line", 1, 1, 8, 0.5, "blue", "wire_top"),
            makeShape("line", 1, 7, 8, 0.5, "blue", "wire_bottom"),
            makeShape("rectangle", 4, 3, 2, 2, "red", "shared_node"),
        ]
    },
    "layout_hint": {
        name: "Layout Preview",
        shapes: [
            makeShape("rectangle", 0.5, 0.5, 3, 2, "green", "block_A"),
            makeShape("rectangle", 4, 0.5, 3, 2, "green", "block_B"),
            makeShape("rectangle", 7.5, 0.5, 2, 2, "purple", "block_C"),
            makeShape("line", 2, 2.5, 0.5, 2, "blue", "via_1"),
            makeShape("line", 5.5, 2.5, 0.5, 2, "blue", "via_2"),
            makeShape("rectangle", 0.5, 4.5, 9, 0.5, "blue", "metal_bus"),
            makeShape("rectangle", 1, 5.5, 3, 2, "teal", "block_D"),
            makeShape("rectangle", 5, 5.5, 4, 2, "orange", "block_E"),
            makeShape("rectangle", 0.5, 8, 9, 0.5, "purple", "metal_bus_2"),
        ]
    }
};

// Also expose as a bundle for backward compat
window.ShapeData = { SHAPE_TYPES, LAYER_COLORS, SHAPE_PARTS, SHAPE_DECOMPOSITION, SAMPLE_SCENES };
