import numpy as np
from numpy.typing import NDArray
from typing import Callable


def gini(y: NDArray):
    """
    Gini = 1 - sum(prob.^2 of each label)
    """
    label_cnts = np.bincount(y)
    label_probs = label_cnts / len(y)
    return 1 - np.sum(label_probs ** 2)


def information_gain(entropy_f: Callable[[NDArray], float], *split_labels: NDArray):
    """
    Args:
        split_labels: labels of each split, shaped `[n_i]`
    """
    total_labels    = np.concat(split_labels, axis=0)
    n_total_labels  = total_labels.shape[0]

    parent_entropy      = entropy_f(total_labels)
    children_entropy    = [entropy_f(v) for v in split_labels]
    children_weight     = [len(v) / n_total_labels for v in split_labels]

    new_entropy = sum([w * e for w, e in zip(children_weight, children_entropy)])
    gain = parent_entropy - new_entropy
    return gain