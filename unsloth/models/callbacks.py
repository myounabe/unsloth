"""Training callbacks for Unsloth fine-tuning runs."""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class TrainingMetrics:
    """Accumulated metrics tracked during training."""
    step: int = 0
    epoch: float = 0.0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    elapsed_seconds: float = 0.0
    samples_per_second: Optional[float] = None
    history: list = field(default_factory=list)

    def record(self, step: int, epoch: float, loss: float, lr: float, elapsed: float, samples_per_sec: Optional[float] = None):
        self.step = step
        self.epoch = epoch
        self.loss = loss
        self.learning_rate = lr
        self.elapsed_seconds = elapsed
        self.samples_per_second = samples_per_sec
        self.history.append({
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "learning_rate": lr,
            "elapsed_seconds": elapsed,
            "samples_per_second": samples_per_sec,
        })

    def best_loss(self) -> Optional[float]:
        """Return the lowest recorded loss, or None if no records exist."""
        losses = [r["loss"] for r in self.history if r["loss"] is not None]
        return min(losses) if losses else None

    def summary(self) -> dict:
        return {
            "total_steps": self.step,
            "final_epoch": self.epoch,
            "final_loss": self.loss,
            "best_loss": self.best_loss(),
            "elapsed_seconds": self.elapsed_seconds,
        }


class UnslothProgressCallback:
    """Lightweight callback that records training progress into a TrainingMetrics object."""

    def __init__(self, verbose: bool = True):
        self.metrics = TrainingMetrics()
        self.verbose = verbose
        self._start_time: Optional[float] = None

    def on_train_begin(self):
        self._start_time = time.monotonic()
        if self.verbose:
            print("[Unsloth] Training started.")

    def on_log(self, step: int, epoch: float, logs: dict):
        elapsed = time.monotonic() - self._start_time if self._start_time is not None else 0.0
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        sps = logs.get("samples_per_second")
        self.metrics.record(step=step, epoch=epoch, loss=loss, lr=lr, elapsed=elapsed, samples_per_sec=sps)
        if self.verbose:
            print(f"[Unsloth] step={step} epoch={epoch:.2f} loss={loss} lr={lr}")

    def on_train_end(self):
        summary = self.metrics.summary()
        if self.verbose:
            print(f"[Unsloth] Training complete. Summary: {summary}")
        return summary
