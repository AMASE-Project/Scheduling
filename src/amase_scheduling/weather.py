import numpy as np


class WeatherModel:
    """Bernoulli per-night clear/cloudy model.

    Each night is independently fully clear with probability `clear_prob`,
    otherwise the entire night is lost to weather.
    """

    def __init__(self, clear_prob: float = 0.5, seed: int | None = None):
        if not 0.0 <= clear_prob <= 1.0:
            raise ValueError(f"clear_prob must be in [0, 1], got {clear_prob}")
        self.clear_prob = clear_prob
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def is_clear(self) -> bool:
        return bool(self._rng.random() < self.clear_prob)
