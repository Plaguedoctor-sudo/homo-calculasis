from dataclasses import dataclass

L_MAX = 1.0


@dataclass
class Load:
    """Allostatic load. Monotone non-decreasing within an episode.

    Axiom A3. Saturation prevents runaway; non-reversibility is asserted,
    not merely observed.
    """

    L: float = 0.0
    decay: float = 0.0          # 0.0 == irreversible within episode

    def integrate(self, deficit: float, dt: float) -> float:
        if deficit < 0.0:
            raise ValueError("deficit must be >= 0")
        head = max(0.0, 1.0 - self.L / L_MAX)
        self.L += deficit * dt * head
        self.L = min(self.L, L_MAX)
        return self.L

    def relax(self, dt: float) -> float:
        """Only called between episodes, never mid-episode."""
        self.L = max(0.0, self.L - self.decay * dt)
        return self.L

    def capacity(self, floor: float = 0.4) -> float:
        """Recovery throughput factor in [floor, 1]. Load slows recovery."""
        return max(floor, 1.0 - 0.6 * self.L)

    def barrier_gain(self, lam: float = 0.8) -> float:
        return lam * self.L
