"""CPU侧预算控制与阶段日程，可在无CUDA环境验收。"""
from dataclasses import dataclass, field
import math

LEVELS = (4000, 3000, 2000, 1000)
DUTIES = (0., .2, .4, .6, .8, 0.)
STAGE_SECONDS = 5.


def stage_at(elapsed):
    if elapsed < 0 or elapsed >= len(DUTIES) * STAGE_SECONDS:
        return None
    return int(elapsed // STAGE_SECONDS)


@dataclass
class BudgetController:
    target_ms: float
    index: int = 0
    low_windows: int = 0
    samples: list = field(default_factory=list)

    def __post_init__(self):
        if not math.isfinite(self.target_ms) or self.target_ms <= 0:
            raise ValueError('Target time must be finite and positive')

    @property
    def budget(self):
        return LEVELS[self.index]

    def update(self, total_ms):
        if not math.isfinite(total_ms) or total_ms <= 0:
            raise ValueError('Frame time must be finite and positive')
        self.samples.append(total_ms)
        if len(self.samples) < 10:
            return
        mean = sum(self.samples) / len(self.samples)
        self.samples.clear()
        if mean > self.target_ms:
            self.index = min(self.index + 1, len(LEVELS) - 1)
            self.low_windows = 0
        elif mean < .85 * self.target_ms:
            self.low_windows += 1
            if self.low_windows >= 2:
                self.index = max(0, self.index - 1)
                self.low_windows = 0
        else:
            self.low_windows = 0
