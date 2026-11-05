from __future__ import annotations
from dataclasses import dataclass

@dataclass
class MitochondrialBioenergetics:
    """Picard Mitochondrial Allostatic Load (MAL): ATP capacity, ROS leak, and mtDNA stress."""
    ros_load: float = 0.2
    mtdna_damage: float = 0.0      # cumulative mitochondrial allostatic wear in [0, 0.8]
    atp_flux: float = 1.0

    def step(self, dt: float, glucose: float, ketone_bhb: float,
             insulin_resistance: float, cortisol: float,
             il6: float, gpx_activity: float) -> float:
        """Updates ROS, mtDNA wear, and net neuronal/somatic ATP synthesis."""
        glycolytic = glucose / (0.5 * insulin_resistance + glucose)
        ketogenic = ketone_bhb / (0.4 + ketone_bhb)  # bypasses insulin resistance
        substrate_drive = min(1.4, glycolytic + ketogenic)

        ros_prod = 0.15 + 0.25 * cortisol + 0.25 * il6 + 0.15 * max(0.0, insulin_resistance - 1.0)
        ros_clear = (0.3 + 0.9 * gpx_activity) * self.ros_load
        self.ros_load = max(0.0, self.ros_load + dt * (ros_prod - ros_clear))

        # Excess unbuffered ROS accumulates irreversible mtDNA wear within episode
        if self.ros_load > 0.5:
            self.mtdna_damage = min(0.8, self.mtdna_damage + dt * 0.01 * (self.ros_load - 0.5))

        self.atp_flux = max(0.1, substrate_drive * (1.0 - self.mtdna_damage))
        return float(self.atp_flux)
