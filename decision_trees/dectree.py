import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable

import sys; sys.path.append(".")
from base_model import BaseModel
from decision_trees.dectree_node import DecTreeNode
from decision_trees.utils import make_tree, lookup_label, report_tree
from decision_trees.infogain_funcs import gini


class DecTree(BaseModel):
    def __init__(
            self,
            max_depth: Optional[int] = 100,
            min_sample_count_per_node: int = 50,
            entropy_func: Callable[[NDArray], float] = gini,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.node_min_sample_count = min_sample_count_per_node
        self.entropy_func = entropy_func

        self.root: Optional[DecTreeNode] = None
    
    def _trained(self):
        return self.root is not None
    
    def train(self, x: NDArray, y: NDArray, **kwargs):
        """
        Args:
            x: shape `[N, D]`
            y: shape `[N]`
        """
        if self.trained:
            cmd = input("Model already trained. Continue? y/[n]")
            if cmd != "y":
                return
        
        import time
        s = time.time()
        self.root = make_tree(
            x, y, 0,
            max_depth=self.max_depth,
            min_sample_count=self.node_min_sample_count,
            entropy_func=self.entropy_func,
        )
        t = time.time()

        report_tree()
        print(f"Time consumption: {(t - s):.2f}s")
    
    def predict(self, x: NDArray, **kwargs):
        """
        Args:
            x: shape `[M, D]`
        Returns:
            y: shape `[M]`
        """
        if not self.trained:
            print(f"{self.__class__.__name__} model not trained yet")
            return
        
        y = [lookup_label(self.root, x_row) for x_row in x]
        return np.array(y)