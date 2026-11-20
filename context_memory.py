"""Interpolated continuation counts: a discrete analogue of prefix/continuation recall.

This is a statistical context model, not a learned Transformer induction head.
Every counter and mixture weight is updated only after the byte is decoded.
"""

import math

import numpy as np

# Унаследованные пороги исходной n-gram реализации, без отдельного подбора.
# Эти же значения используются в fast_memory.cpp; округление сохраняет счётчики > 0.
GLOBAL_RESCALE_THRESHOLD = 32768
CONTEXT_RESCALE_THRESHOLD = 16384


class ContinuationMemory:
    def __init__(
        self,
        vocab_size,
        batch_size,
        order=6,
        initial_neural_weight=0.5,
        rate=0.01,
        escape_scale=1.0,
    ):
        self.vocab_size = vocab_size
        self.order = order
        self.history = [b"" for _ in range(batch_size)]
        self.global_counts = np.ones(vocab_size, dtype=np.float64)
        self.global_total = float(vocab_size)
        self.tables = {}
        self.initial_neural_weight = initial_neural_weight
        self.log_odds = math.log(initial_neural_weight / (1 - initial_neural_weight))
        self.rate = rate
        self.escape_scale = escape_scale
        self.mixture_predictions = 0

    def predict(self, stream):
        """Interpolate suffixes from short to long using already observed bytes."""
        probabilities = self.global_counts / self.global_total
        history = self.history[stream]
        for length in range(1, min(self.order, len(history)) + 1):
            entry = self.tables.get(history[-length:])
            if entry is None:
                continue
            counts, total = entry
            escape = self.escape_scale * len(counts)
            divisor = total + escape
            probabilities = probabilities * (escape / divisor)
            for symbol, count in counts.items():
                probabilities[symbol] += count / divisor
        return probabilities

    def weight(self):
        return 1 / (1 + math.exp(-self.log_odds))

    def combine(self, neural, memory):
        weight = self.weight()
        return weight * neural + (1 - weight) * memory

    def observe(self, stream, symbol, neural_probability=None, memory_probability=None):
        """Update statistics after coding/decoding, identically on both sides."""
        if neural_probability is not None:
            weight = self.weight()
            neural_probability = float(neural_probability)
            memory_probability = float(memory_probability)
            mixture = max(weight * neural_probability + (1 - weight) * memory_probability, 1e-12)
            # Gradient of log P_mix(actual_byte), rather than picking one expert.
            gradient = weight * (1 - weight) * (neural_probability - memory_probability) / mixture
            self.log_odds = max(-8.0, min(8.0, self.log_odds + self.rate * gradient))
            self.mixture_predictions += 1
        self.global_counts[symbol] += 1
        self.global_total += 1
        if self.global_total >= GLOBAL_RESCALE_THRESHOLD:
            self.global_counts = np.maximum(1, np.ceil(self.global_counts / 2))
            self.global_total = self.global_counts.sum()
        history = self.history[stream]
        for length in range(1, min(self.order, len(history)) + 1):
            context = history[-length:]
            entry = self.tables.get(context)
            if entry is None:
                entry = self.tables[context] = [{}, 0]
            counts, total = entry
            counts[symbol] = counts.get(symbol, 0) + 1
            total += 1
            if total >= CONTEXT_RESCALE_THRESHOLD:
                counts = {key: (count + 1) // 2 for key, count in counts.items()}
                total = sum(counts.values())
            entry[0], entry[1] = counts, total
        if self.order:
            self.history[stream] = (history + bytes((symbol,)))[-self.order :]

    def stats(self):
        return {
            "context_tables": len(self.tables),
            "neural_weight_final": self.weight(),
            "mixture_predictions": self.mixture_predictions,
        }
