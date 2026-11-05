from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TripartiteSynapse:
    """Astrocytic EAAT2 glutamate clearance, extrasynaptic spillover, and BDNF modulation."""
    glu_extrasynaptic: float = 0.2
    bdnf: float = 1.0
    k_eaat2_base: float = 1.5

    def step(self, dt: float, synaptic_drive: float, qa: float, il6: float,
             gs_activity: float, atp_flux: float, mg_free: float = 1.0) -> None:
        release = synaptic_drive * (1.0 + 0.35 * qa)
        eaat2_cap = self.k_eaat2_base * atp_flux * (0.3 + 0.7 * gs_activity) / (
            1.0 + 0.5 * il6 + 0.4 * qa
        )
        self.glu_extrasynaptic += dt * (release - eaat2_cap * self.glu_extrasynaptic)
        self.glu_extrasynaptic = max(0.05, self.glu_extrasynaptic)

        # Extrasynaptic NMDA activation (unblocked when Mg is low) suppresses BDNF
        excitotoxic_drive = (self.glu_extrasynaptic + 0.4 * qa) / (0.3 + mg_free)
        bdnf_target = 1.2 / (0.6 + excitotoxic_drive)
        self.bdnf += dt * (bdnf_target - self.bdnf) / 4.0
        self.bdnf = max(0.1, self.bdnf)

    @property
    def ei_ratio_shift(self) -> float:
        """Maps extrasynaptic glutamate spillover to cortical E/I branching ratio (> 1.0)."""
        return float(1.0 + 0.4 * max(0.0, self.glu_extrasynaptic - 0.25))
