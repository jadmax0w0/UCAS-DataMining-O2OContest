import numpy as np
from numpy.typing import NDArray
from typing import Optional, Any

class DecTreeNode():
    def __init__(
            self,
            left: Optional[Any] = None,
            right: Optional[Any] = None,
            splitter_feat_dim: Optional[int] = None,
            splitter_feat_val: Optional[float] = None,
            *,
            label_value: Optional[Any] = None,
    ):
        self.left = left
        self.right = right

        self.splitter_feat_dim = splitter_feat_dim
        self.splitter_feat_val = splitter_feat_val
        
        self.label_value = label_value
    
    @property
    def is_leaf(self):
        return self.label_value is not None
    
    def next_node(self, x: NDArray):
        """
        Args:
            x: shaped `[D]`
        """
        assert len(x.shape) == 1, f"{x.shape=}"

        if self.is_leaf:
            return None
        
        if x[self.splitter_feat_dim] < self.splitter_feat_val:
            return self.left
        else:
            return self.right