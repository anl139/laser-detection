import numpy as np


class TemporalHeatmapFilter:

    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.prev = None

    def reset(self):
        self.prev = None

    def update(self, heatmap):

        if self.prev is None:
            self.prev = heatmap.copy()
            return heatmap

        filtered = (
            self.alpha * self.prev
            +
            (1.0 - self.alpha) * heatmap
        )

        self.prev = filtered.copy()

        return filtered