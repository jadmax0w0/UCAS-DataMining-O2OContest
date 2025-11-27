import numpy as np
import time
from tqdm import tqdm
from numpy.typing import NDArray
from typing import Optional, Union, List, Literal
from base_model import BaseModel


class SimpleLinearSVM(BaseModel):
    def __init__(
            self,
            lr: float = 0.001,
            lambda_value: float = 0.01,
            # hidden_him: Optional[int] = None,
            **kwargs,
    ):
        self.lr = lr
        self.lambda_value = lambda_value
        # self.hidden_dim = hidden_him

        self.w = None
        self.b = None
    
    def _trained(self):
        return self.w is not None and self.b is not None
    
    @staticmethod
    def _loss(x: NDArray, y: NDArray, w: NDArray, b: NDArray, lambda_value: float):
        N = x.shape[0]
        
        distances = 1 - y * (np.dot(x, w) + b)
        
        distances[distances < 0] = 0
        hinge_loss = np.sum(distances) / N
        
        loss = (lambda_value / 2) * np.dot(w, w) + hinge_loss
        return loss
    
    def train(
            self,
            x: NDArray,
            y: NDArray,
            epochs: int = 1000,
            loss_path: Optional[str] = None,
    ):
        """
        Args:
            x: shape `[N, D]`
            y: shape `[N]`
        """
        def save_loss(loss_path, epoch, x, y, w, b, l):
            if loss_path is not None: # and (epoch % 10 == 0 or epoch < 0):  # save loss every 10 epochs
                epoch_loss = SimpleLinearSVM._loss(x, y, w, b, l)

                import os
                fdir, _ = os.path.split(loss_path)
                if fdir != "":
                    os.makedirs(fdir, exist_ok=True)
                with open(loss_path, mode=('a' if os.path.exists(loss_path) else 'w'), encoding='utf-8') as f:
                    f.write(f"Epoch {epoch}: {epoch_loss}\n")
        
        N, D = x.shape

        y = np.where(y <= 0, -1, 1)
        assert y.shape[0] == N, f"{y.shape[0]=}"

        if self.trained:
            cmd = input("Model already trained. Continue? y/[n]")
            if cmd != "y":
                return
        
        s = time.time()

        w = np.zeros(D)
        b = 0
        save_loss(loss_path, -1, x, y, w, b, self.lambda_value)

        for epoch in tqdm(range(epochs), desc="Epoch", total=epochs):
            for idx, xi in enumerate(x):
                yi = y[idx]

                correct = yi * (np.dot(xi, w) + b) >= 1
                if correct:
                    w -= self.lr * (2 * self.lambda_value * w)
                else:
                    w -= self.lr * (2 * self.lambda_value * w - np.dot(xi, yi))
                    b -= self.lr * (-yi)
            
            save_loss(loss_path, epoch, x, y, w, b, self.lambda_value)
        
        self.w = w
        self.b = b

        t = time.time()
        print(f"Training complete. Time consumption: {(t - s):.2f}s")
    
    def predict(self, x: Union[List[NDArray], NDArray], raw_output: bool = False):
        """
        Args:
            x: shape `[n, D]`
        Returns:
            y (NDArray): shape `[n]`
        """
        out = np.dot(x, self.w) + self.b
        return out if raw_output else np.sign(out)
