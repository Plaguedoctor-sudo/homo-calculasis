#!/usr/bin/env python3
from __future__ import annotations
import csv
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Auto-re-execute using repository .venv if running with an interpreter missing dependencies
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    try:
        import numpy
    except ImportError:
        if __name__ == "__main__":
            os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__] + sys.argv[1:])
        else:
            raise

import numpy as np
from core.whole_body_runner import WholeBodyPatient
from inference.design import select_optimal_clamp
from inference.hierarchical import fit_hierarchical_eb
from inference.npe_train import train_amortized_npe
from inference.simulator import simulate_batch
from params.clamps import evaluate_clamped_observables
from params.space import ParameterSpace
from validation.prospective import evaluate_prospective_trial, predict_arm_probabilities
from validation.recovery import evaluate_recovery

HTML_FILE = REPO_ROOT / "ui" / "mega_chart.html"

# Pre-train amortized NPE engine for live sub-second inferences
space = ParameterSpace()
train_thetas = space.sample_prior(1000, seed=42)
train_xs = simulate_batch(train_thetas, space, noise_sd=0.015, seed=43)
npe_engine = train_amortized_npe(train_thetas, train_xs, space)

# Cache last simulation trajectory
last_trajectory: list[dict[str, float]] = []

class ClinicalServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            if not HTML_FILE.exists():
                self.send_error(404, "Dashboard HTML not found")
                return
            content = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            return
        if path in ("/api/health", "/api/validation", "/api/biotypes", "/api/export"):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        self.send_error(404, "Not Found")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            if not HTML_FILE.exists():
                self.send_error(404, "Dashboard HTML not found")
                return
            content = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/health":
            self.send_json_response({
                "status": "active",
                "simulator": "WholeBodyPatient",
                "dim": space.dim,
                "endpoints": [
                    "/api/health", "/api/patient/infer", "/api/simulate",
                    "/api/clamp", "/api/design", "/api/validation",
                    "/api/biotypes", "/api/export"
                ]
            })
            return

        if path == "/api/validation":
            # Run synthetic EMBARC trial validation & T1 recovery
            test_thetas = space.sample_prior(200, seed=22)
            test_xs = simulate_batch(test_thetas, space, noise_sd=0.015, seed=23)
            rec = evaluate_recovery(npe_engine, test_thetas, test_xs)

            cohort_true = space.sample_prior(250, seed=70)
            cohort_xs = simulate_batch(cohort_true, space, noise_sd=0.015, seed=71)
            cohort_inf, _ = npe_engine.predict_posterior(cohort_xs)
            prospective = evaluate_prospective_trial(cohort_true, cohort_inf, space, seed=72)

            self.send_json_response({
                "recovery_r2_mean": round(rec.mean_r2, 4),
                "recovery_r2_per_param": {k: round(v, 3) for k, v in rec.r2_per_param.items()},
                "auc_sertraline": round(prospective.auc_sertraline, 3),
                "calibration_slope": round(prospective.calibration_slope, 3),
                "calibration_intercept": round(prospective.calibration_intercept, 3),
                "drug_placebo_divergence": round(prospective.drug_placebo_divergence, 3),
                "t1_pass": rec.mean_r2 >= 0.75,
                "t3_pass": prospective.auc_sertraline >= 0.60 and (0.6 <= prospective.calibration_slope <= 1.4),
                "t4_pass": prospective.drug_placebo_divergence >= 0.10,
            })
            return

        if path == "/api/biotypes":
            # Hierarchical clustering / Empirical Bayes subtype extraction
            b1 = space.sample_prior(80, seed=50)
            b1[:, space.names.index("ido_gain")] = 2.6
            b2 = space.sample_prior(80, seed=51)
            b2[:, space.names.index("reward_rho")] = 0.35
            b2[:, space.names.index("fe_intake")] = 0.25
            cohort_thetas = np.vstack([b1, b2])

            cohort_xs = simulate_batch(cohort_thetas, space, noise_sd=0.015, seed=52)
            pat_means, post_cov = npe_engine.predict_posterior(cohort_xs)
            hier = fit_hierarchical_eb(pat_means, post_cov)

            top_axes = []
            for j in range(min(3, hier.eigenvectors.shape[1])):
                axis = hier.eigenvectors[:, j]
                top_indices = np.argsort(np.abs(axis))[::-1][:3]
                top_params = [f"{space.names[idx]} ({axis[idx]:+.2f})" for idx in top_indices]
                top_axes.append({
                    "axis": j + 1,
                    "variance_pct": round(float(hier.eigenvalues[j] / np.sum(hier.eigenvalues) * 100), 1),
                    "leading_biomarkers": top_params
                })

            self.send_json_response({
                "effective_rank": round(hier.effective_rank, 2),
                "subtype_axes": top_axes,
                "biotypes_confirmed": hier.effective_rank > 1.5,
            })
            return

        if path == "/api/export":
            global last_trajectory
            if not last_trajectory:
                self.send_error(400, "No simulation has been run yet")
                return

            buf = io.StringIO()
            fieldnames = ["t_hours", "m", "load_L", "qa_ka", "il6", "serotonin", "dopamine", "atp"]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for row in last_trajectory:
                writer.writerow(row)

            csv_bytes = buf.getvalue().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", 'attachment; filename="patient_trajectory.csv"')
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        payload = json.loads(post_data.decode("utf-8")) if post_data else {}

        if path == "/api/patient/infer":
            # Live Bayesian parameter inversion via Amortized NPE
            profile = payload.get("profile", "indian_veg_anhedonia")
            theta_true = space.mean.copy()

            if profile in ("indian_veg_anhedonia", "anhedonic_iron"):
                theta_true[space.names.index("fe_intake")] = 0.18
                theta_true[space.names.index("zn_intake")] = 0.55
                theta_true[space.names.index("reward_rho")] = 0.28
                theta_true[space.names.index("social_support")] = 0.40
            elif profile in ("south_asian_metabolic", "metabolic_glial"):
                theta_true[space.names.index("insulin_resistance")] = 4.2
                theta_true[space.names.index("eaat2_capacity")] = 0.40
                theta_true[space.names.index("ido_gain")] = 1.9
            elif profile == "inflammatory_kynurenine":
                theta_true[space.names.index("ido_gain")] = 2.8
                theta_true[space.names.index("scfa_production")] = 0.22
                theta_true[space.names.index("vagal_tone")] = 0.30
            elif profile in ("melancholic_hpa_failure", "psychosocial_defeat"):
                theta_true[space.names.index("social_support")] = 0.15
                theta_true[space.names.index("barrier_lam")] = 1.8
                theta_true[space.names.index("pi_prior")] = 3.0
                theta_true[space.names.index("vagal_tone")] = 0.25
            elif profile == "euthymic_healthy":
                theta_true[space.names.index("fe_intake")] = 1.05
                theta_true[space.names.index("zn_intake")] = 1.0
                theta_true[space.names.index("social_support")] = 0.85
                theta_true[space.names.index("reward_rho")] = 0.85

            x_obs = simulate_batch(theta_true[None, :], space, noise_sd=0.01, seed=123)[0]
            inf_mean, post_cov = npe_engine.predict_posterior(x_obs)
            contractions = 1.0 - np.diag(post_cov) / (space.sd ** 2)

            params_dict = {}
            for name, val_inf, val_true, cont in zip(space.names, inf_mean, theta_true, contractions):
                params_dict[name] = {
                    "inferred": round(float(val_inf), 3),
                    "true": round(float(val_true), 3),
                    "contraction": round(float(cont), 3),
                }

            self.send_json_response({
                "profile": profile,
                "parameters": params_dict,
                "readout_summary": {
                    "il6": round(float(x_obs[6]), 3),
                    "qa_ka": round(float(x_obs[7]), 3),
                    "serotonin": round(float(x_obs[4]), 3),
                    "prt_bias": round(float(x_obs[5]), 3),
                    "dnic": round(float(x_obs[8]), 3),
                }
            })
            return

        if path == "/api/simulate":
            global last_trajectory
            patient = WholeBodyPatient()

            # Indian Clinical Laboratory Parameters Mapping
            # 1. Hematology & Iron Studies (NABL standards)
            ferr = float(payload.get("ferritin", 18.0))
            patient.axis.cof.fe.intake = max(0.05, ferr / 60.0)

            # 2. Vitamins & Trace Elements (Indian clinical units)
            zn_val = float(payload.get("zinc", 55.0))  # µg/dL (Ref 70-120)
            patient.axis.cof.zn.intake = max(0.05, zn_val / 80.0)

            se_val = float(payload.get("selenium", 65.0))  # µg/L (Ref 70-150)
            patient.axis.cof.se.intake = max(0.05, se_val / 100.0)

            mg_val = float(payload.get("magnesium", 1.7))  # mg/dL (Ref 1.8-2.4)
            patient.axis.cof.mg.intake = max(0.05, mg_val / 2.2)

            b12_val = float(payload.get("b12", 150.0))  # pg/mL (Ref 211-911; <200 deficient)
            vitd_val = float(payload.get("vit_d", 16.0))  # ng/mL (Ref 30-100; <20 deficient)

            # 3. South Asian Glycemic & Insulin Profile (ICMR / RSSDI)
            fbs_val = float(payload.get("fbs", 112.0))  # mg/dL (Ref 70-99)
            insulin_val = float(payload.get("fasting_insulin", 18.0))  # µIU/mL (Ref 2.6-24.9)
            if "homa_ir" in payload:
                homa_ir_val = float(payload["homa_ir"])
            else:
                homa_ir_val = (fbs_val * insulin_val) / 405.0  # Standard HOMA-IR formula
            patient.macro.insulin_resistance = max(0.5, homa_ir_val)

            # 4. Inflammatory & Immune Markers
            il6_val = float(payload.get("il6", 4.5))  # pg/mL (Ref < 5.0)
            hscrp_val = float(payload.get("hs_crp", 3.2))  # mg/L (Ref < 1.0)
            patient.axis.cyto.il6 = max(0.1, (il6_val / 2.0) + (hscrp_val / 4.0))

            # 5. Neuroendocrine, Thyroid & Stress Axis
            tsh_val = float(payload.get("tsh", 3.8))  # µIU/mL (Ref 0.4-4.5)
            ft3_val = float(payload.get("ft3", 2.4))  # pg/mL (Ref 2.0-4.4)
            cortisol_am = float(payload.get("cortisol_am", 22.0))  # µg/dL (Ref 6-23)
            dex_post = float(payload.get("dex_suppression", 3.2))  # µg/dL (Ref < 1.8 normal suppression)
            patient.hpt.t3_brain = max(0.1, (ft3_val / 3.0) * (2.0 / max(0.5, tsh_val)))

            # 6. Autonomic Vagal & Enteric Permeability
            rmssd_val = float(payload.get("hrv_rmssd", 22.0))  # ms (Ref 35-70)
            patient.axis.vagal.efferent = max(0.1, rmssd_val / 40.0)

            lps_val = float(payload.get("lps", 0.6))  # EU/mL (Ref < 0.2)
            patient.axis.barrier.lps_lumen = max(0.1, lps_val * 4.0)
            patient.env.pm25_ug_m3 = max(5.0, 5.0 + lps_val * 20.0)

            patient.psy.relational_support = float(payload.get("social_support", 0.4))
            patient.m = float(payload.get("initial_m", -0.85))

            # Additional penalty if B12 is severely deficient (< 200 pg/mL)
            b12_depletion_factor = 0.55 if b12_val < 200 else (0.80 if b12_val < 300 else 1.0)
            patient.axis.trp.serotonin *= b12_depletion_factor

            # HPA non-suppression (DEX >= 1.8 µg/dL) increases baseline stress strain
            if dex_post >= 1.8:
                patient.load.L += 0.25 * (dex_post - 1.8)

            # 7. Apply User-Created / Custom Parameters
            custom_params = payload.get("custom_parameters", [])
            for cp in custom_params:
                try:
                    c_val = float(cp.get("value", 1.0))
                    c_base = float(cp.get("baseline", c_val))
                    c_min = float(cp.get("min", 0.0))
                    c_max = float(cp.get("max", 10.0))
                    c_target = cp.get("target_axis", "il6_inflammation")
                    c_dir = cp.get("direction", "positive")
                    span = max(0.001, c_max - c_min)
                    norm_delta = (c_val - c_base) / span
                    mult_factor = max(0.05, 1.0 + (norm_delta if c_dir == "positive" else -norm_delta))

                    if c_target == "il6_inflammation":
                        patient.axis.cyto.il6 = max(0.05, patient.axis.cyto.il6 * mult_factor)
                    elif c_target == "serotonin_monoamine":
                        patient.axis.trp.serotonin = max(0.05, patient.axis.trp.serotonin * mult_factor)
                    elif c_target == "dopamine_synthesis":
                        patient.axis.cof.fe.intake = max(0.05, patient.axis.cof.fe.intake * mult_factor)
                    elif c_target == "eaat2_clearance":
                        patient.glia.k_eaat2_base = max(0.1, patient.glia.k_eaat2_base * mult_factor)
                    elif c_target == "mito_atp":
                        eff = norm_delta if c_dir == "positive" else -norm_delta
                        patient.macro.ketone_bhb = max(0.0, patient.macro.ketone_bhb + eff * 2.5)
                    elif c_target == "allostatic_load":
                        eff = norm_delta if c_dir == "positive" else -norm_delta
                        patient.load.L = max(0.0, patient.load.L + eff * 0.8)
                    elif c_target == "vagal_tone":
                        patient.axis.vagal.efferent = max(0.05, patient.axis.vagal.efferent * mult_factor)
                    elif c_target == "gut_barrier":
                        patient.axis.barrier.lps_lumen = max(0.05, patient.axis.barrier.lps_lumen * mult_factor)
                    elif c_target == "mood_m":
                        eff = norm_delta if c_dir == "positive" else -norm_delta
                        patient.m = max(-1.2, min(1.2, patient.m + eff * 0.6))
                    elif c_target == "thyroid_t3":
                        patient.hpt.t3_brain = max(0.05, patient.hpt.t3_brain * mult_factor)
                except Exception:
                    continue

            intervention = payload.get("intervention", "none")
            if intervention == "repletion":
                patient.axis.cof.fe.intake = max(patient.axis.cof.fe.intake, 2.0)
                patient.axis.cof.zn.intake = max(patient.axis.cof.zn.intake, 2.0)
                patient.axis.cof.se.intake = max(patient.axis.cof.se.intake, 2.0)
                patient.axis.cof.mg.intake = max(patient.axis.cof.mg.intake, 1.8)
                patient.axis.trp.serotonin /= b12_depletion_factor  # Repletion restores B12
            elif intervention == "ssri":
                patient.axis.trp.serotonin *= 1.6
            elif intervention == "rtms":
                patient.glia.k_eaat2_base = 2.2
                patient.axis.vagal.efferent *= 1.5
            elif intervention == "ketogenic":
                patient.macro.ketone_bhb = 2.5
            elif intervention == "full_stack":
                patient.axis.cof.fe.intake = max(patient.axis.cof.fe.intake, 2.0)
                patient.axis.cof.zn.intake = max(patient.axis.cof.zn.intake, 2.0)
                patient.axis.cof.se.intake = max(patient.axis.cof.se.intake, 2.0)
                patient.axis.cof.mg.intake = max(patient.axis.cof.mg.intake, 1.8)
                patient.axis.trp.serotonin /= b12_depletion_factor
                patient.axis.trp.serotonin *= 1.6
                patient.glia.k_eaat2_base = 2.2
                patient.macro.ketone_bhb = 2.5
                patient.env.lux = 10000.0
                patient.psy.relational_support = 1.0

            dt = 0.05
            steps = int(payload.get("steps", 1440))
            record_interval = max(1, steps // 60)

            traj = []
            for step in range(steps):
                patient.step(dt=dt)
                if step % record_interval == 0:
                    traj.append({
                        "t_hours": round(step * dt, 2),
                        "m": round(float(patient.m), 4),
                        "load_L": round(float(patient.load.L), 4),
                        "qa_ka": round(float(patient.axis.trp.qa_ka), 4),
                        "il6": round(float(patient.axis.cyto.il6), 4),
                        "serotonin": round(float(patient.axis.trp.serotonin), 4),
                        "dopamine": round(float(patient.axis.cof.outputs()["dopamine_synthesis"]), 4),
                        "atp": round(float(patient.mito.atp_flux), 4),
                    })

            last_trajectory = traj
            basin = "Resilient Euthymic Well (+1.0)" if patient.m > 0 else "Depressive Sinking Well (-1.0)"

            # Compute Doctor-Friendly Symptom Scales (0-10)
            final_da = float(patient.axis.cof.outputs()["dopamine_synthesis"])
            final_5ht = float(patient.axis.trp.serotonin)
            final_atp = float(patient.mito.atp_flux)
            final_qa_ka = float(patient.axis.trp.qa_ka)
            final_il6 = float(patient.axis.cyto.il6)
            final_m = float(patient.m)
            final_L = float(patient.load.L)

            anhedonia = round(min(10.0, max(0.0, 10.0 * (1.0 - min(1.0, final_da * 1.1)))), 1)
            fatigue = round(min(10.0, max(0.0, 10.0 * (1.0 - min(1.0, 0.55 * final_atp + 0.45 * patient.hpt.t3_brain)))), 1)
            brain_fog = round(min(10.0, max(0.0, 0.45 * fatigue + 0.35 * anhedonia + 0.2 * min(10.0, final_qa_ka * 2.2))), 1)
            dnic_gate = float(patient.axis.vagal.efferent) / (1.0 + 0.4 * final_qa_ka)
            somatic_pain = round(min(10.0, max(0.0, 10.0 * (1.0 - min(1.0, dnic_gate * 0.95)) + 0.35 * final_il6)), 1)
            sleep_disruption = round(min(10.0, max(0.0, 10.0 * (1.0 - min(1.0, float(patient.sleep.sws_density) * 1.3)))), 1)

            # Sinking risk percentage (0 - 100%)
            sinking_risk = round(min(100.0, max(0.0, (1.0 - (final_m + 1.0) / 2.0) * 85.0 + 15.0 * min(2.0, final_L))), 1)

            # Treatment Response Probabilities (%) with Indian Clinical Factors
            fe_supp = min(1.0, patient.axis.cof.fe.intake / 1.0)
            b12_penalty = 0.45 if b12_val < 200 else (0.75 if b12_val < 300 else 1.0)
            dex_penalty = 0.70 if dex_post >= 1.8 else 1.0
            ssri_eff = round(min(88.0, max(6.0, (75.0 * fe_supp * b12_penalty * dex_penalty) / (1.0 + 0.25 * max(0.0, final_il6 - 1.2)))), 1)
            rtms_eff = round(min(88.0, max(28.0, 68.0 + (16.0 if ssri_eff < 40.0 else 0.0) - 4.0 * min(2.0, final_L))), 1)
            ketamine_eff = round(min(94.0, max(30.0, 48.0 + 14.0 * min(3.0, final_qa_ka) + (8.0 if dex_post >= 1.8 else 0.0))), 1)
            trace_eff = round(min(95.0, max(20.0, 88.0 if (ferr < 30 or b12_val < 250 or zn_val < 70 or se_val < 70) else 35.0)), 1)

            # Clinical Directives & Doctor-Facing Alerts (Indian Practice Guidelines)
            clinical_alerts = []
            if ferr < 30.0:
                clinical_alerts.append(f"🩸 Serum Ferritin is critically low ({ferr:.0f} ng/mL, Ref: 30–300 ng/mL): TPH2 serotonin synthesis is stalled. SSRI monotherapy has high failure rate. Co-prescribe Ferrous Ascorbate 100mg elemental iron OD (or IV Ferric Carboxymaltose if ferritin < 15 ng/mL).")
            if b12_val < 211.0:
                clinical_alerts.append(f"💊 Severe Vitamin B12 Deficiency ({b12_val:.0f} pg/mL, Ref: 211–911 pg/mL): Common in Indian vegetarian diets. Methylation cycle failure impairs SAMe and BH4 neurotransmitter synthesis. Prescribe Methylcobalamin 1500 mcg OD + L-Methylfolate 7.5 mg.")
            if vitd_val < 20.0:
                clinical_alerts.append(f"☀️ Severe Vitamin D3 Deficiency ({vitd_val:.1f} ng/mL, Ref: 30–100 ng/mL): Impairs neurotrophin (BDNF) transcription and worsens chronic neuroinflammation. Prescribe Cholecalciferol 60,000 IU weekly for 8 weeks.")
            if homa_ir_val > 2.5:
                clinical_alerts.append(f"🥩 South Asian Metabolic Insulin Resistance (HOMA-IR: {homa_ir_val:.2f}, Ref: < 1.5; FBS: {fbs_val:.0f} mg/dL): Astrocytic bioenergetic starvation impairs EAAT2 glutamate clearance. Consider Metformin 500mg BD / Myo-Inositol and low-glycemic dietary restriction.")
            if hscrp_val > 3.0 or il6_val > 5.0:
                clinical_alerts.append(f"🛡️ High-Grade Systemic Inflammation (hs-CRP: {hscrp_val:.1f} mg/L, Ref: < 1.0; IL-6: {il6_val:.1f} pg/mL): IDO1 enzyme active, diverting tryptophan away from 5-HT toward neurotoxic Quinolinic Acid.")
            if final_qa_ka > 3.2:
                clinical_alerts.append(f"⚠️ Elevated QA/KA Excitotoxicity Ratio ({final_qa_ka:.2f}, Ref: < 3.0): Astrocytic EAAT2 glutamate transporters overwhelmed. SSRIs often provoke agitation. Prioritize Left DLPFC 10Hz rTMS or NMDA Receptor Antagonists (Ketamine / Esketamine) + Zinc Bisglycinate 50mg HS.")
            if dex_post >= 1.8:
                clinical_alerts.append(f"🚨 Overnight 1mg DEX Non-Suppression ({dex_post:.1f} µg/dL, Ref: < 1.8 µg/dL): Severe HPA axis glucocorticoid receptor resistance indicative of endogenous melancholic MDD with elevated relapse risk.")
            if not clinical_alerts:
                clinical_alerts.append("✅ Physiological biomarkers within compensated euthymic ranges. Good neurotrophic resilience.")

            # Synthesize Doctor's Formal Consultation & Prescription Slip
            prescriptions = []
            if ferr < 30.0:
                prescriptions.append({"item": "Tab. Ferrous Ascorbate (100mg Elemental Fe) + Folic Acid", "dose": "1 tab OD after lunch with Vitamin C", "duration": "3 months", "notes": "Target Ferritin > 50 ng/mL"})
            if b12_val < 300.0:
                prescriptions.append({"item": "Tab. Methylcobalamin 1500 mcg + L-Methylfolate 7.5 mg + P-5-P 50 mg", "dose": "1 tab OD morning after breakfast", "duration": "3 months", "notes": "Restores SAMe & BH4 synthesis"})
            if vitd_val < 30.0:
                prescriptions.append({"item": "Syp / Cap. Cholecalciferol (Vitamin D3) 60,000 IU", "dose": "1 dose weekly with milk", "duration": "8 weeks, then monthly", "notes": "Target 25-OH Vit D > 40 ng/mL"})
            if zn_val < 70.0 or final_qa_ka > 3.0:
                prescriptions.append({"item": "Tab. Zinc Bisglycinate 50 mg + Magnesium Glycinate 250 mg", "dose": "1 tab at bedtime (HS)", "duration": "2 months", "notes": "Co-activates KAT; NMDA channel blocker"})
            if homa_ir_val > 2.5:
                prescriptions.append({"item": "Tab. Metformin SR 500 mg (or Myo-Inositol 2000 mg)", "dose": "1 tab OD post-dinner", "duration": "Continuous", "notes": "Improves brain insulin sensitivity"})
            if ssri_eff >= 45.0:
                prescriptions.append({"item": "Tab. Escitalopram 10 mg (or Sertraline 50 mg)", "dose": "1 tab OD morning", "duration": "Review after 4 weeks", "notes": "Initiate only alongside cofactor repletion"})
            elif rtms_eff >= 65.0:
                prescriptions.append({"item": "Left Dorsolateral Prefrontal Cortex (DLPFC) 10 Hz rTMS Protocol", "dose": "30 sessions (3000 pulses/session, 120% MT)", "duration": "6 weeks (5 days/wk)", "notes": "Upregulates astrocytic EAAT2, bypasses synthesis bottleneck"})

            doctor_rx = {
                "patient_type": "Indian Clinical Presentation",
                "homa_ir": round(homa_ir_val, 2),
                "qa_ka_ratio": round(final_qa_ka, 2),
                "dex_status": "Non-Suppressor (Melancholic Risk)" if dex_post >= 1.8 else "Normal Suppression",
                "recommended_investigations": [
                    "Complete Blood Count (CBC) with Peripheral Smear (NABL)",
                    "Comprehensive Iron Profile: Serum Ferritin + Serum Iron + TIBC + % Saturation",
                    "Serum Vitamin B12 (Active Cyanocobalamin) & Serum Folate",
                    "25-Hydroxy Vitamin D3 [25-OH D3]",
                    "Fasting Blood Sugar (FBS) & Fasting Serum Insulin (HOMA-IR calculation)",
                    "Thyroid Profile: TSH, Free T3 (FT3), Free T4 (FT4)",
                    "High-Sensitivity C-Reactive Protein (hs-CRP) & IL-6",
                    "Overnight 1mg Dexamethasone Suppression Test (8 AM post-DEX Cortisol)"
                ],
                "prescriptions": prescriptions
            }

            downstream_cascade = {
                "dopamine_synthesis": round(final_da, 3),
                "serotonin_flux": round(final_5ht, 3),
                "atp_bioenergetics": round(final_atp, 3),
                "qa_ka_ratio": round(final_qa_ka, 3),
                "allostatic_load_L": round(final_L, 3),
                "mood_basin_m": round(final_m, 3),
                "homa_ir": round(homa_ir_val, 2),
                "vagal_efferent": round(float(patient.axis.vagal.efferent), 3),
                "cytokine_il6": round(final_il6, 3),
                "dnic_pain_gate": round(float(dnic_gate), 3),
                "sleep_sws_density": round(float(patient.sleep.sws_density), 3),
                "eaat2_clearance": round(float(patient.glia.k_eaat2_base), 3),
            }

            self.send_json_response({
                "final_m": float(patient.m),
                "final_L": float(patient.load.L),
                "final_qa_ka": float(patient.axis.trp.qa_ka),
                "basin_status": basin,
                "downstream_cascade": downstream_cascade,
                "symptoms": {
                    "anhedonia": anhedonia,
                    "fatigue": fatigue,
                    "brain_fog": brain_fog,
                    "somatic_pain": somatic_pain,
                    "sleep_disruption": sleep_disruption,
                    "sinking_risk_pct": sinking_risk,
                },
                "treatment_predictions": {
                    "ssri_pct": ssri_eff,
                    "rtms_pct": rtms_eff,
                    "ketamine_pct": ketamine_eff,
                    "trace_repletion_pct": trace_eff,
                },
                "clinical_alerts": clinical_alerts,
                "doctor_rx": doctor_rx,
                "trajectory": traj,
            })
            return

        if path == "/api/clamp":
            active = payload.get("active_clamps", None)
            if active is not None:
                active = tuple(active)

            theta = space.sample_prior(1, seed=42)[0]
            x_obs = evaluate_clamped_observables(theta, space, active_clamps=active, noise_sd=0.0)
            obs_names = [
                "fe_obs", "zn_obs", "mg_obs", "se_obs",
                "serotonin_5ht", "prt_bias", "il6_level", "qa_ka_ratio",
                "dnic", "pain_report", "homa_ir_obs", "lat1_trp_influx",
                "metabolite_load", "glu_spillover", "t3_brain",
                "hb_acc", "open_hidden_gap", "barrier_depth",
                "social_eval"
            ]
            readout = {name: round(float(val), 4) for name, val in zip(obs_names, x_obs)}
            self.send_json_response({"active_clamps": active, "readouts": readout})
            return

        if path == "/api/design":
            # Active experiment selection via Bayesian Optimal Experimental Design (BOED)
            applied = tuple(payload.get("already_applied", ["trace_panel"]))
            best_clamp, gains = select_optimal_clamp(space, already_applied=applied, n_sim=150, seed=60)
            self.send_json_response({
                "already_applied": list(applied),
                "best_clamp": best_clamp,
                "information_gains_nats": {k: round(v, 3) for k, v in gains.items()},
            })
            return

        self.send_error(404, "Not Found")


def run_server(host: str = "0.0.0.0", port: int = 8050):
    ThreadingHTTPServer.allow_reuse_address = True
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, ClinicalServerHandler)
    print(f"\n=======================================================")
    print(f"🚀 Whole-Body Clinical Workstation Live on Localhost!")
    print(f"👉 Web Browser URL: http://localhost:{port}")
    print(f"👉 API Endpoints: /api/simulate, /api/patient/infer,")
    print(f"                 /api/clamp, /api/design, /api/validation,")
    print(f"                 /api/biotypes, /api/export")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nWorkstation server terminated.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    host = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"
    run_server(host=host, port=port)
