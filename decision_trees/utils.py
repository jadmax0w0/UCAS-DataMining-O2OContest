import numpy as np
from numpy.typing import NDArray
import sys; sys.path.append(".")
from decision_trees.dectree_node import DecTreeNode
from decision_trees.infogain_funcs import information_gain, gini
from typing import Callable
from tqdm import tqdm


def most_frequent_elem(x: NDArray):
    """
    Returns:
        `most_frequent_value` and its `count` (tuple form)
    """
    x = x.flatten()
    values, counts = np.unique(x, return_counts=True)
    hot_id = np.argmax(counts)
    return values[hot_id], counts[hot_id]


_make_tree_call_count = 0

def make_tree(
        x: NDArray,
        y: NDArray,
        curr_depth: int = 0,
        *,
        max_depth: int = 100,
        min_sample_count: int = 2,
        entropy_func: Callable[[NDArray], float] = gini,
):
    """
    Args:
        x: shape `[n, D]`
        y: shape `[n]`
    """
    global _make_tree_call_count
    _make_tree_call_count += 1
    CC = _make_tree_call_count
    
    n, D = x.shape
    labels_cnt = len(np.unique(y))

    # Early exit
    # 1. max depth; 2. too few samples for this node; 3. only 1 label presents in this node
    if max_depth is None:
        max_depth = float('inf')
    if curr_depth >= max_depth or n <= min_sample_count or labels_cnt <= 1:
        # Majority vote
        freq_elem, _ = most_frequent_elem(y)
        # print(f"Leaf node {CC} created (early exit)")
        return DecTreeNode(label_value=freq_elem)
    
    # Get the best feature dimension and value for splitting
    best_gain = float("-inf")
    best_feat_dim = None
    best_feat_value = None
    
    # for fdim in tqdm(range(D), desc=f"Checking feature dimensions for node {CC}", total=D):
    for fdim in range(D):
        x_projed = x[:, fdim]  # [n]
        feat_values = np.unique(x_projed)  # 这里如果用的是 PCA 降维过的，feat_values 会很多（理论上与样本个数相等）；但如果用原始特征则顶多 255 个
        
        # Track the best information gain when split by each feature value
        for fval in feat_values:
            if fval == feat_values[0] or fval == feat_values[-1]:
                continue
            # Try splitting using this feature value
            left_indices = np.argwhere(x_projed < fval).flatten()
            right_indices = np.argwhere(x_projed >= fval).flatten()
            # Get information gain
            gain = information_gain(entropy_func, y[left_indices], y[right_indices])
            
            if gain > best_gain:
                best_gain = gain
                best_feat_dim = fdim
                best_feat_value = fval
    
    if best_gain == float("-inf") or best_feat_dim is None or best_feat_value is None:
        # Failed in locating the best feature dim for splitting
        # e.g. 所有 sample 在所有剩余 feature dim 上的 value 都相同
        freq_elem, _ = most_frequent_elem(y)
        # print(f"Leaf node {CC} craeted")
        return DecTreeNode(label_value=freq_elem)

    # SPLIT!
    left_indices = np.argwhere(x[:, best_feat_dim] < best_feat_value).flatten()
    right_indices = np.argwhere(x[:, best_feat_dim] >= best_feat_value).flatten()

    left_tree = make_tree(
        x[left_indices], y[left_indices], curr_depth + 1,
        max_depth=max_depth,
        min_sample_count=min_sample_count,
        entropy_func=entropy_func
    )
    right_tree = make_tree(
        x[right_indices], y[right_indices], curr_depth + 1,
        max_depth=max_depth,
        min_sample_count=min_sample_count,
        entropy_func=entropy_func
    )

    # print(f"Trunk node {CC} craeted")
    return DecTreeNode(
        left=left_tree,
        right=right_tree,
        splitter_feat_dim=best_feat_dim,
        splitter_feat_val=best_feat_value,
    )


def report_tree():
    global _make_tree_call_count
    print(f"*** Created {_make_tree_call_count} nodes for the latest tree")
    _make_tree_call_count = 0


def lookup_label(tree_node: DecTreeNode, x: NDArray):
    """
    Args:
        x: a single vector for a sample, shape `[D]`
    """
    assert len(x.shape) == 1, f"{x.shape=}"

    if tree_node is None:
        return None
    
    node = tree_node
    while True:
        next_node = node.next_node(x)
        if next_node is None:  # `node` is a leaf node
            return node.label_value
        node = next_node