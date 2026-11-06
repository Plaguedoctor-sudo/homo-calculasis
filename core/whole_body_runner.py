# core/whole_body_runner.py
from __future__ import annotations
from dataclasses import dataclass, field
from cofactors import enzyme_capacity as ec
from endocrine.axes import HPTAxis
from environment.exposome import Exposome
from immune.axis import ImmuneAxis
from load import Load
from memory.sleep import SleepGlymphatic
from mood.landscape import Params, dU
from neuroglia.astrocyte import TripartiteSynapse
from nutrition.bioenergetics import MitochondrialBioenergetics
from nutrition.macronutrients import MacronutrientState
from psychosocial.models import PsychosocialState


@dataclass
class WholeBodyPatient:
    axis: ImmuneAxis = field(default_factory=ImmuneAxis)
    env: Exposome = field(default_factory=Exposome)
    macro: MacronutrientState = field(default_factory=MacronutrientState)
    mito: MitochondrialBioenergetics = field(default_factory=MitochondrialBioenergetics)
    psy: PsychosocialState = field(default_factory=PsychosocialState)
    glia: TripartiteSynapse = field(default_factory=TripartiteSynapse)
    hpt: HPTAxis = field(default_factory=HPTAxis)
    sleep: SleepGlymphatic = field(default_factory=SleepGlymphatic)
    load: Load = field(default_factory=Load)
    mood_params: Params = field(default_factory=Params)
    m: float = 1.0                 # start in healthy well (+1.0)
    mn_free: float = 1.0           # manganese pool for glutamine synthetase

    def step(self, dt: float = 0.05) -> None:
        # 1. Environment & Psychosocial -> Immune & HPA
        self.axis.cortisol = (
            0.4 * self.psy.social_baseline_multiplier
            + 0.3 * self.psy.bandwidth_tax
            + self.env.thermal_allostatic_cost
        )
        self.axis.cyto.lps_challenge = (
            self.env.pm25_inflammatory_drive + self.psy.ctra_inflammatory_drive
        )

        # 2. Macronutrient LAT1 competition -> Brain Tryptophan uptake
        self.axis.trp.k_uptake = 0.4 * self.macro.bbb_trp_uptake_factor(self.axis.trp.trp_plasma)

        # 3. Step gut-immune-cofactor axis (modulated by omega-3 resolution gain)
        self.axis.vagal.efferent *= self.macro.spm_resolution_gain
        self.axis.step(dt)

        # 4. Mitochondrial bioenergetics & HPT thyroid axis
        out = self.axis.cof.outputs()
        atp = self.mito.step(
            dt, glucose=self.macro.carb_intake, ketone_bhb=self.macro.ketone_bhb,
            insulin_resistance=self.macro.insulin_resistance, cortisol=self.axis.cortisol,
            il6=self.axis.cyto.il6, gpx_activity=out["gpx_activity"],
        )
        self.hpt.step(dt, se_free=self.axis.cof.se.free(), cortisol=self.axis.cortisol, il6=self.axis.cyto.il6)

        # 5. Neuro-glial tripartite synapse & Sleep/Glymphatic clearance
        gs_act = ec.capacity(ec.GS, self.mn_free)
        self.glia.step(
            dt, synaptic_drive=1.0, qa=self.axis.trp.qa, il6=self.axis.cyto.il6,
            gs_activity=gs_act, atp_flux=atp, mg_free=self.axis.cof.mg.free(),
        )
        sws_eff = 0.7 * self.env.sws_noise_penalty / (1.0 + 0.3 * self.axis.cortisol)
        self.sleep.step(dt, sws_density=sws_eff)

        # 6. Allostatic load integration & Mood landscape evolution
        deficit = (
            0.05 * self.env.thermal_allostatic_cost
            + 0.05 * self.psy.bandwidth_tax
            + 0.05 * self.mito.mtdna_damage
            + 0.05 * max(0.0, self.glia.glu_extrasynaptic - 0.25)
        )
        self.load.integrate(deficit, dt)

        # Positive mood support from 5-HT, DA*reward_gain, T3, BDNF, photoperiod; negative from Joiner + QA/KA
        da_eff = self.psy.effective_reward_sensitivity(out["dopamine_synthesis"])
        trophic_support = 0.25 * (
            self.axis.trp.serotonin * self.env.photoperiod_drive
            + da_eff + self.hpt.t3_brain + self.glia.bdnf
        )
        allostatic_stress = (
            self.psy.interpersonal_mood_drift
            + 0.15 * self.axis.trp.qa_ka
            + 0.20 * self.psy.bandwidth_tax
        )
        net_drive = allostatic_stress - trophic_support + 0.2
        barrier = self.load.barrier_gain(self.mood_params.lam)
        self.m -= dU(self.m, net_drive, barrier, self.mood_params.a, self.mood_params.b) * dt
