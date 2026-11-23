import os
import copy
from typing import Dict, Optional
import torch

class EarlyStopping:
    """
    Decentralized Early Stopping tracker.
    Monitors validation performance (e.g., validation accuracy or loss) across rounds.
    Stops training when no improvement is observed for `patience` rounds,
    and preserves the best performing model weights.
    """
    def __init__(
        self,
        patience: int = 4,
        min_delta: float = 0.001,
        mode: str = "max",
        checkpoint_path: str = "checkpoints/best_model.pt"
    ):
        """
        Args:
            patience: number of rounds to wait for improvement before stopping.
            min_delta: minimum change to qualify as an improvement.
            mode: 'max' (for accuracy) or 'min' (for loss).
            checkpoint_path: file path to persist the best model weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        
        self.best_score: Optional[float] = None
        self.best_round: int = 0
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        self.counter: int = 0
        self.early_stop: bool = False

    def is_better(self, current: float, best: float) -> bool:
        if self.mode == "max":
            return current > (best + self.min_delta)
        else:
            return current < (best - self.min_delta)

    def step(
        self,
        metric: float,
        round_num: int,
        model_weights: Dict[str, torch.Tensor]
    ) -> bool:
        """
        Evaluates the metric for round_num and updates state.
        Returns:
            is_new_best: True if current metric improved upon best_score.
        """
        if self.best_score is None:
            self.best_score = metric
            self.best_round = round_num
            self._save_checkpoint(model_weights, metric, round_num)
            return True

        if self.is_better(metric, self.best_score):
            self.best_score = metric
            self.best_round = round_num
            self.counter = 0
            self._save_checkpoint(model_weights, metric, round_num)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def _save_checkpoint(self, weights: Dict[str, torch.Tensor], metric: float, round_num: int):
        self.best_weights = {k: v.cpu().clone() for k, v in weights.items()}
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        torch.save({
            "round": round_num,
            "metric": metric,
            "state_dict": self.best_weights
        }, self.checkpoint_path)

    def load_best_weights(self) -> Dict[str, torch.Tensor]:
        """Loads and returns the best saved weights."""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            return checkpoint["state_dict"]
        elif self.best_weights is not None:
            return self.best_weights
        else:
            raise FileNotFoundError("No best checkpoint found.")
