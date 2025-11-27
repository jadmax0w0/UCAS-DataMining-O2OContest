import numpy as np
from numpy.typing import NDArray
import sys; sys.path.append(".")
from base_model import BaseModel
from svms.linear_svm import SimpleLinearSVM
from typing import List, Dict, Optional


class MultiClassSVM(BaseModel):
    """
    One-vs-Rest
    """
    def __init__(
            self,
            lr: float = 0.001,
            svm_class: type = SimpleLinearSVM,
            **svmkwargs
    ):
        super().__init__()

        self.lr = lr

        self.svm_class = svm_class

        self.classifiers: Optional[List[int, BaseModel]] = None
        self.classes = None

        self.svm_kwargs = svmkwargs
    
    def _trained(self):
        return self.classes is not None and self.classifiers is not None and all([v.trained for v in self.classifiers])
    
    def train(self, x: NDArray, y: NDArray, epochs: int = 1000, **kwargs):
        if self.trained:
            cmd = input("Model already trained. Continue? y/[n]")
            if cmd != "y":
                return
        
        self.classes = np.unique(y)
        self.classifiers = []

        for c in self.classes:
            print(f"Training SVM for class {c}")
            svm = self.svm_class(lr=self.lr, **self.svm_kwargs)
            
            y_ = np.where(y == c, 1, -1)
            svm.train(x, y_, epochs, **kwargs)

            self.classifiers.append(svm)
        
        print(f"Training complete for all {len(self.classes)} classes")
    
    def predict(self, x: NDArray, raw_output = False, **kwargs):
        """
        Args:
            x: shape `[n, D]`
        Returns:
            scores shaped `[n, class_cnt]` if `raw_output`, else labels shaped `[n]`
        """
        n, D = x.shape
        C = len(self.classes)

        scores = np.zeros((n, C))

        for k, classifer in enumerate(self.classifiers):
            scores[:, k] = classifer.predict(x, raw_output=True, **kwargs)
        
        return scores if raw_output else self.classes[np.argmax(scores, axis=-1)]
