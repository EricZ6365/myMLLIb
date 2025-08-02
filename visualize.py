import matplotlib.pyplot as plt
from collections import defaultdict
_op_color = {
    "Leaf": "green",
    "MatMulOp": "#FF5733",
    "SumOp": "#3357FF",
    "MeanOp": "#3357FF",
    "MaxOp": "#3357FF",
    "NegOp": "#3357FF",

    "ExpOp": "#FF33A6",
    "SubOp": "#FF33A6",
    "AddOp": "#FF33A6",
    "MulOp": "#FF33A6",
    "DivOp": "#FF33A6",

    "AbsOp": "#FF33A6",
    "PowOp": "#FF33A6",
    "LogOp": "#FF33A6",
    "LnOp": "#FF33A6",
    "ClampOp": "#A633FF",

    "BroadCastOp": "#33FF8F",
    "IndexOp": "#33A6FF",
    "UnsqueezeOp": "#33A6FF",
    "SqueezeOp": "#33A6FF",
    "FlattenOp": "#33A6FF",

    "ConcatOp": "#8F33FF"

}
class _DrawGraphNode:
    def __init__(self, op_name):
        self.op_name = op_name

    def __repr__(self):
        return f"{self.op_name}"


def visualize_DCG(root, figsize=(8, 6), node_size=1000, font_size=7):
    memo = {}
    edges = set()
    visited = set()
    depth = {}

    def dfs_build(node, d=0):
        if node not in memo:
            memo[node] = _DrawGraphNode(node.op.__class__.__name__ if node.op is not None else "Leaf")
        depth[memo[node]] = max(depth.get(memo[node], -1), d)

        if node in visited:
            return
        visited.add(node)

        for parent in getattr(node, "children", []) or []:
            if not parent:
                continue
            if parent not in memo:
                memo[parent] = _DrawGraphNode(parent.op.__class__.__name__ if parent.op is not None else "Leaf")
            edges.add((memo[parent], memo[node]))
            dfs_build(parent, d + 1)

    dfs_build(root.grad_node, d=0)

    by_depth = defaultdict(list)
    for n, d in depth.items():
        by_depth[d].append(n)

    positions = {}
    for d in sorted(by_depth.keys()):
        layer = by_depth[d]
        layer.sort(key=lambda n: (n.op_name, id(n)))
        n_in_layer = len(layer)
        for i, n in enumerate(layer):
            x = d
            y = i - (n_in_layer - 1) / 2.0
            positions[n] = (x, y)

    max_d = max(depth.values())
    for n in positions:
        x, y = positions[n]
        positions[n] = (x, max_d - y)

    plt.figure(figsize=figsize)

    for parent, child in edges:
        x1, y1 = positions[parent]
        x2, y2 = positions[child]
        plt.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.2,
                connectionstyle="arc3,rad=0.2"
            )
        )

    for n, (x, y) in positions.items():
        color = _op_color[n.op_name]
        plt.scatter([x], [y], s=node_size, color=color, edgecolors='black', linewidths=0.8)
        plt.text(x, y, n.op_name, ha='center', va='center', fontsize=font_size, color='black')

    plt.axis('off')
    plt.title("Dynamic Computational Graph")
    plt.tight_layout()
    plt.show()
