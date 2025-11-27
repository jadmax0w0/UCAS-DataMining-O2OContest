import numpy as np
from numpy.typing import NDArray
from typing import Any

import sys; sys.path.append(".")
from base_model import BaseModel


class VotingBoost(BaseModel):
    def __init__(
            self,
            *models: BaseModel,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.models = models
    
    def _trained(self):
        return all([m.trained for m in self.models])
    
    @staticmethod
    def _majority_vote(*y_preds: NDArray):
        """
        Args:
            y_preds: shape `[N]`
        Returns:
            modes of `y_preds`, shape `[N]`
        """
        from scipy import stats
        
        stacked_preds = np.stack(y_preds, axis=0)
        mode_result = stats.mode(stacked_preds, axis=0, keepdims=False)

        if hasattr(mode_result, 'mode'):
            # Old scipy versions
            return mode_result.mode if np.ndim(mode_result.mode) == stacked_preds.ndim -1 else mode_result.mode[0]
        
        return mode_result[0]
    
    @property
    def use_soft_output(self):
        # 涉及到树一类的模型都没法取到预测的软标签，先不弄这个了
        return NotImplementedError()
    
    def train(self, x: NDArray, y: NDArray, epochs: Any = None, **kwargs):
        """
        Args:
            x: shape `[N, D]`
            y: shape `[N]`
        """
        import time
        for idx, model in enumerate(self.models):
            print(f"Training model {idx}")
            s = time.time()
            try:
                model.train(x, y, epochs=epochs, **kwargs)
            except AttributeError:
                model.fit(x, y)
            t = time.time()
            print(f"Model {idx} trained, time consumption: {(t-s):.2f}s")
    
    def predict(self, x: NDArray, **kwargs):
        """
        Args:
            x: shape `[n, D]`
        Returns:
            y: voted result across all models, shape `[n]`
        """
        # if not self.trained:
        #     print("Not all models are trained while boosting, train them first")
        #     return
        
        results = [model.predict(x, **kwargs) for model in self.models]
        return VotingBoost._majority_vote(*results)
