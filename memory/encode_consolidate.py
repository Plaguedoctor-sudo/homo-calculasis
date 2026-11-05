from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemorySystem:
    hippocampal_trace: float = 0.5
    cortical_store: float = 0.5

    def step_encode_consolidate(self, novelty: float, sws_density: float,
                                metabolite_load: float, dt: float = 0.1) -> None:
        encode_eff = 1.0 / (1.0 + 0.2 * metabolite_load)
        self.hippocampal_trace += dt * (encode_eff * novelty - 0.1 * self.hippocampal_trace)
        transfer = 0.3 * sws_density * self.hippocampal_trace
        self.cortical_store += dt * transfer

    def retrieval_accuracy(self, metabolite_load: float, cortisol: float = 0.5) -> tuple[float, float]:
        """Metabolite accumulation degrades retrieval more steeply than encoding."""
        encode_acc = 1.0 / (1.0 + 0.2 * metabolite_load)
        retrieve_acc = (self.cortical_store / (0.3 + self.cortical_store)) / (
            1.0 + 1.2 * metabolite_load + 0.4 * cortisol
        )
        return float(encode_acc), float(retrieve_acc)
